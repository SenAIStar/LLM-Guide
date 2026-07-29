from __future__ import annotations


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def citation_metrics(claims: list[dict]) -> dict[str, float]:
    cited = [claim for claim in claims if claim["citation_ids"]]
    supported = [claim for claim in cited if claim["citations_support_claim"]]
    needs_evidence = [claim for claim in claims if claim["needs_evidence"]]
    covered = [claim for claim in needs_evidence if claim["citation_ids"]]
    return {
        "citation_precision": safe_rate(len(supported), len(cited)),
        "citation_completeness": safe_rate(len(covered), len(needs_evidence)),
        "unsupported_claim_rate": safe_rate(
            sum(
                claim["needs_evidence"] and not claim["citations_support_claim"]
                for claim in claims
            ),
            len(needs_evidence),
        ),
    }
