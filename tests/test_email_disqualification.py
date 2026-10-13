import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent as agent_module


class ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {
            "prospect_id": "LEAD-TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }
        self.rep = {"name": "Ava Mills", "email": "ava@example.com"}

    def send(self, *, source_disqualified, passed_disqualified):
        self.prospect["disqualified"] = passed_disqualified
        with patch.object(
            agent_module.data_service,
            "get_prospect_record",
            return_value={"disqualified": source_disqualified},
        ):
            return agent_module.send_prospect_email.func(
                self.prospect,
                "A quick introduction",
                "Hello there",
                self.runtime,
                self.rep,
            )

    def test_source_record_blocks_stale_passed_prospect(self):
        result = self.send(source_disqualified=True, passed_disqualified=False)

        self.assertEqual(
            result,
            {
                "status": "blocked",
                "error": "Prospect is disqualified; email not sent.",
                "to": "prospect@example.com",
            },
        )
        self.assertNotIn("message_id", result)

    def test_passed_prospect_blocks_when_source_record_is_not_disqualified(self):
        result = self.send(source_disqualified=False, passed_disqualified=True)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["error"], "Prospect is disqualified; email not sent.")
        self.assertNotIn("message_id", result)

    def test_qualified_prospect_is_sent(self):
        result = self.send(source_disqualified=False, passed_disqualified=False)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "prospect@example.com")
        self.assertIn("message_id", result)


class AgentEmailResponseTests(unittest.TestCase):
    def invoke_with_send_result(self, *, source_disqualified, passed_disqualified, final_reply):
        prospect = {
            "prospect_id": "LEAD-TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": passed_disqualified,
        }
        model = ScriptedChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_current_rep", "args": {}, "id": "rep-call", "type": "tool_call"}
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "send_prospect_email",
                                "args": {
                                    "prospect": prospect,
                                    "subject": "A quick introduction",
                                    "body": "Hello there",
                                    "from_rep": {"name": "Ava Mills", "email": "ava@example.com"},
                                },
                                "id": "email-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content=final_reply),
                ]
            )
        )
        agent = create_agent(
            model=model,
            tools=[agent_module.get_current_rep, agent_module.send_prospect_email],
            system_prompt=agent_module.SYSTEM_PROMPT,
        )
        with patch.object(
            agent_module.data_service,
            "get_prospect_record",
            return_value={"disqualified": source_disqualified},
        ):
            return agent.invoke(
                {"messages": [{"role": "user", "content": "Email this prospect."}]},
                config={
                    "configurable": {"thread_id": "test-thread"},
                    "metadata": {"user_id": "rep_amills"},
                },
            )

    def test_agent_explains_disqualified_block(self):
        result = self.invoke_with_send_result(
            source_disqualified=True,
            passed_disqualified=False,
            final_reply="The email was not sent because the prospect is disqualified.",
        )

        tool_result = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == "send_prospect_email"
        )
        self.assertEqual(json.loads(tool_result.content)["status"], "blocked")
        self.assertIn("not sent", result["messages"][-1].content.lower())
        self.assertIn("disqualified", result["messages"][-1].content.lower())

    def test_agent_preserves_successful_send_response(self):
        result = self.invoke_with_send_result(
            source_disqualified=False,
            passed_disqualified=False,
            final_reply="The email was sent successfully.",
        )

        tool_result = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == "send_prospect_email"
        )
        tool_output = json.loads(tool_result.content)
        self.assertEqual(tool_output["status"], "sent")
        self.assertIn("message_id", tool_output)
        self.assertEqual(result["messages"][-1].content, "The email was sent successfully.")


if __name__ == "__main__":
    unittest.main()
