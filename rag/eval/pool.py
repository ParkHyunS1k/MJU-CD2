"""여러 검색 방식의 상위 결과를 합쳐 판정 후보를 만든다(TREC식 pooling)."""


def pool_query(system_rankings: dict[str, list[str]], per_system: int) -> list[dict]:
    pooled: dict[str, dict] = {}
    for rank in range(per_system):
        for system, ranking in system_rankings.items():
            if rank >= len(ranking):
                continue
            entry = pooled.setdefault(ranking[rank], {"content_hash": ranking[rank], "systems": []})
            entry["systems"].append(system)
    return list(pooled.values())
