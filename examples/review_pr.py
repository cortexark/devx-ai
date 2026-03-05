"""Example: Review a pull request using devx-ai.

Usage:
    export DEVX_LLM_API_KEY="sk-..."
    python examples/review_pr.py owner/repo 42

    # Without LLM (AST-only):
    python examples/review_pr.py --ast-only owner/repo 42
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from devx.core.config import GitHubConfig, LLMConfig
from devx.review.agent import ReviewAgent
from devx.review.suggestions import SuggestionFormatter
from devx.sdlc.github_client import GitHubClient


async def review_pr(repo: str, pr_number: int, *, ast_only: bool = False) -> None:
    """Fetch a PR diff and run code review."""

    # 1. Fetch the diff from GitHub
    github_config = GitHubConfig()
    async with GitHubClient(github_config) as gh:
        print(f"Fetching diff for {repo}#{pr_number}...")
        diff_text = await gh.get_pull_request_diff(repo, pr_number)

    # 2. Configure the review agent
    if ast_only:
        agent = ReviewAgent(enable_llm=False)
        print("Running AST-only review (no LLM)...")
    else:
        llm_config = LLMConfig()
        agent = ReviewAgent(llm_config=llm_config, enable_llm=True)
        print("Running AST + LLM review...")

    # 3. Run the review
    result = await agent.review_diff(diff_text)

    # 4. Display results
    formatter = SuggestionFormatter()
    formatter.print_terminal(result)

    # 5. Also generate GitHub comment markdown
    comment = formatter.to_github_comment(result)
    print("\n--- GitHub Comment Preview ---")
    print(comment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a GitHub PR with devx-ai")
    parser.add_argument("repo", help="Repository in owner/repo format")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument(
        "--ast-only",
        action="store_true",
        help="Run AST analysis only (no LLM)",
    )
    args = parser.parse_args()

    asyncio.run(review_pr(args.repo, args.pr_number, ast_only=args.ast_only))


if __name__ == "__main__":
    main()
