import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import (
    SCORE_WEIGHTS,
    SCORING_PROMPT,
    ProspectScore,
    RubricBreakdown,
    score_prospect,
)


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Test offering",
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 1,
        }

    def _result(self, llm_score, revenue_fit, tech_stack_match, segment_fit, justification="Fit"):
        return ProspectScore(
            score=llm_score,
            justification=justification,
            rubric_breakdown=RubricBreakdown(
                revenue_fit=revenue_fit,
                tech_stack_match=tech_stack_match,
                segment_fit=segment_fit,
            ),
        )

    def _score(self, result):
        with patch("gtm_agent.gtm_agent._scoring_llm", Mock(invoke=Mock(return_value=result))):
            return score_prospect.func({"prospect_id": None}, self.offering)

    def test_components_100_75_100_produce_weighted_score(self):
        result = self._result(99, 100, 75, 100)

        scored = self._score(result)

        self.assertEqual(scored["score"], 87.5)
        self.assertEqual(scored["justification"], "Fit")

    def test_component_changes_update_score_using_documented_weights(self):
        scores = [
            self._score(self._result(0, 100, 75, 100))["score"],
            self._score(self._result(0, 100, 70, 100))["score"],
            self._score(self._result(0, 100, 75, 80))["score"],
        ]

        self.assertEqual(scores, [87.5, 85.0, 83.5])
        self.assertEqual(SCORE_WEIGHTS, {
            "revenue_fit": 0.3,
            "tech_stack_match": 0.5,
            "segment_fit": 0.2,
        })
        self.assertIn("revenue_fit (30%)", SCORING_PROMPT)
        self.assertIn("tech_stack_match (50%)", SCORING_PROMPT)
        self.assertIn("segment_fit (20%)", SCORING_PROMPT)

    def test_same_components_ignore_different_llm_scores(self):
        with patch(
            "gtm_agent.gtm_agent._scoring_llm",
            Mock(invoke=Mock(side_effect=[
                self._result(90, 100, 75, 100, "First justification"),
                self._result(85, 100, 75, 100, "Second justification"),
            ])),
        ):
            first = score_prospect.func({"prospect_id": None}, self.offering)
            second = score_prospect.func({"prospect_id": None}, self.offering)

        self.assertEqual(first["score"], 87.5)
        self.assertEqual(second["score"], 87.5)
        self.assertEqual(first["justification"], "First justification")
        self.assertEqual(second["justification"], "Second justification")
        self.assertEqual(set(first), {"score", "max_score", "justification", "rubric_breakdown"})


if __name__ == "__main__":
    unittest.main()
