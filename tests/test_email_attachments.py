import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class EmailAttachmentTests(unittest.TestCase):
    def test_document_request_cannot_be_claimed_as_attached(self):
        self.assertIn("do not claim or imply", SYSTEM_PROMPT)
        self.assertIn("attached or included in the email", SYSTEM_PROMPT)

    def test_rep_is_told_document_was_not_included(self):
        self.assertIn("document was not included", SYSTEM_PROMPT)
        self.assertIn("must be sent manually", SYSTEM_PROMPT)

    def test_email_tool_sends_only_plain_text_without_attachment_claim(self):
        description = send_prospect_email.description
        self.assertIn("plain-text email", description)
        self.assertIn("Attachments are not supported", description)

        body = "I'd be happy to discuss pricing with you."
        result = send_prospect_email.func(
            prospect={"name": "Avery Example", "email": "avery@example.com"},
            subject="Pricing discussion",
            body=body,
            runtime=SimpleNamespace(config={}),
            from_rep={"name": "Rep Example", "email": "rep@example.com"},
        )

        self.assertIsInstance(result["body"], str)
        self.assertEqual(result["body"], body)
        self.assertNotIn("attachments", result)


if __name__ == "__main__":
    unittest.main()
