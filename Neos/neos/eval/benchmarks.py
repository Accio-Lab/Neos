
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class Benchmark:
    id: str
    display_name: str
    slug: str
    aliases: tuple[str, ...]
    description: str


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        id="mmsearch",
        display_name="MMSearch",
        slug="mmsearch",
        aliases=("mm-search",),
        description="MMSearch agentic multimodal deep-search evaluation.",
    ),
    Benchmark(
        id="browsecomp_vl",
        display_name="BrowseComp-VL",
        slug="browsecomp-vl",
        aliases=("bcvl", "browsecomp-vl", "browsecompvl"),
        description="BrowseComp-VL agentic multimodal browsing evaluation.",
    ),
    Benchmark(
        id="mm_browsecomp",
        display_name="MM-BrowseComp",
        slug="mm-browsecomp",
        aliases=("mmbrowsecomp", "mm-browsecomp", "mmbrowse_comp"),
        description="Official MM-BrowseComp agentic evaluation.",
    ),
    Benchmark(
        id="mmsearch_plus",
        display_name="MMSearch-Plus",
        slug="mmsearch-plus",
        aliases=("mmsearch-plus", "mmsearchplus", "mm_search_plus"),
        description="Official MMSearch-Plus agentic evaluation.",
    ),
    Benchmark(
        id="simplevqa",
        display_name="SimpleVQA",
        slug="simplevqa",
        aliases=("simple-vqa", "simple_vqa"),
        description="SimpleVQA knowledge-intensive visual QA evaluation.",
    ),
    Benchmark(
        id="livevqa",
        display_name="LiveVQA",
        slug="livevqa",
        aliases=("live-vqa", "live_vqa", "livevqa-research-preview"),
        description="LiveVQA Research Preview live visual-knowledge evaluation.",
    ),
    Benchmark(
        id="fvqa",
        display_name="FVQA-test",
        slug="fvqa",
        aliases=("fvqa-test", "fvqa_test"),
        description="FVQA-test fact-based visual QA evaluation.",
    ),
    Benchmark(
        id="vdr",
        display_name="VDR",
        slug="vdr",
        aliases=("vdr-bench", "vdr_bench", "vision-deepresearch"),
        description="Official VDR-Bench testmini evaluation.",
    ),
    Benchmark(
        id="browsecomp_v3",
        display_name="BrowseComp-V³",
        slug="browsecomp-v3",
        aliases=(
            "bc-v3",
            "bc_v3",
            "bcv3",
            "browsecomp-v3",
            "browsecomp-v³",
            "browsecompv3",
        ),
        description="Official BrowseComp-V³ multimodal browsing evaluation.",
    ),
    Benchmark(
        id="visbrowse",
        display_name="VisBrowse",
        slug="visbrowse",
        aliases=("visbrowse-bench", "visbrowse_bench", "vis-browse"),
        description="Official VisBrowse-Bench visual-native search evaluation.",
    ),
    Benchmark(
        id="infoseek",
        display_name="InfoSeek-MMSearch-R1-2K",
        slug="infoseek-mmsearch-r1-2k",
        aliases=(
            "info-seek",
            "info_seek",
            "infoseek-human",
            "infoseek_human",
            "infoseek-mmsearch-r1",
            "infoseek_mmsearch_r1",
        ),
        description=(
            "Exact released MMSearch-R1 2,000-item sample of the official "
            "InfoSeek test split."
        ),
    ),
)



TEXT_SEARCH_BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark('browsecomp', 'BrowseComp', 'browsecomp', ('browse-comp',),
              'Official English BrowseComp1266 text-input search evaluation.'),
    Benchmark('browsecomp_zh', 'BrowseComp-ZH', 'browsecomp-zh', ('browsecomp-zh',),
              'Official Chinese BrowseComp-ZH289 text-input search evaluation.'),
    Benchmark('hle_text', 'HLE-Text', 'hle-text', ('hle-text', 'hle-text-2158'),
              'HLE2158 no-image questions from the pinned public2500 copy, original question wording.'),
    Benchmark('deepsearchqa', 'DeepSearchQA', 'deepsearchqa', ('deep-search-qa',),
              'Google DeepSearchQA900; VisionDR correctness and separate set-grading macro-F1.'),
)

BENCHMARK_BY_ID = {benchmark.id: benchmark for benchmark in (*BENCHMARKS, *TEXT_SEARCH_BENCHMARKS)}


def _alias_key(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "_")


_ALIASES: dict[str, str] = {}
for _benchmark in (*BENCHMARKS, *TEXT_SEARCH_BENCHMARKS):
    for _alias in (
        _benchmark.id,
        _benchmark.slug,
        _benchmark.display_name,
        *_benchmark.aliases,
    ):
        _ALIASES[_alias_key(_alias)] = _benchmark.id


SELECTION_GROUPS: dict[str, tuple[str, ...]] = {
    "text_search4": ("browsecomp", "browsecomp_zh", "hle_text", "deepsearchqa"),
    "browsecomp_pair": ("browsecomp", "browsecomp_zh"),
    "hle_dsqa": ("hle_text", "deepsearchqa"),
    "both": ("mmsearch", "browsecomp_vl"),
    "new": ("mm_browsecomp", "mmsearch_plus"),
    "remaining": ("browsecomp_vl", "mm_browsecomp", "mmsearch_plus"),
    
    
    "all": ("mmsearch", "browsecomp_vl", "mm_browsecomp", "mmsearch_plus"),
    "extended": (
        "simplevqa",
        "livevqa",
        "fvqa",
        "vdr",
        "browsecomp_v3",
        "visbrowse",
    ),
    "visiondr300": ("simplevqa", "fvqa", "livevqa"),
    "visiondr300_sf": ("simplevqa", "fvqa"),
    "visiondr300_sl": ("simplevqa", "livevqa"),
    "simplevqa300": ("simplevqa",),
    "fvqa300": ("fvqa",),
    "livevqa300_mc": ("livevqa",),
    "vdr_visbrowse": ("vdr", "visbrowse"),
    "vdr_simplevqa300": ("vdr", "simplevqa"),
    "bcv3_mmsearch": ("browsecomp_v3", "mmsearch"),
    "bcvl_mmbc_simplevqa300": ("browsecomp_vl", "mm_browsecomp", "simplevqa"),
    "bcv3_mmbc": ("browsecomp_v3", "mm_browsecomp"),
    "simplevqa300_bcv3_mmbc": ("simplevqa", "browsecomp_v3", "mm_browsecomp"),
    "mmbc_bcvl_bcv3_simplevqa300": (
        "mm_browsecomp", "browsecomp_vl", "browsecomp_v3", "simplevqa",
    ),
    "all10": tuple(
        benchmark.id for benchmark in BENCHMARKS if benchmark.id != "infoseek"
    ),
    "all11": tuple(benchmark.id for benchmark in BENCHMARKS),
}


def canonical_benchmark_id(
    value: str,
    *,
    allow_unknown: bool = False,
) -> str:
    pass
    raw = str(value or "").strip()
    canonical = _ALIASES.get(_alias_key(raw))
    if canonical:
        return canonical
    if allow_unknown:
        return raw or "unknown"
    raise ValueError(
        f"unsupported benchmark {value!r}; expected one of "
        + ", ".join(benchmark.display_name for benchmark in BENCHMARKS)
    )


def canonical_benchmark_selection(value: str) -> str:
    key = _alias_key(value)
    if key in SELECTION_GROUPS:
        return key
    return canonical_benchmark_id(value)


def expand_benchmark_selection(value: str) -> tuple[str, ...]:
    canonical = canonical_benchmark_selection(value)
    return SELECTION_GROUPS.get(canonical, (canonical,))


def benchmark_display_name(value: str) -> str:
    canonical = canonical_benchmark_id(value, allow_unknown=True)
    benchmark = BENCHMARK_BY_ID.get(canonical)
    return benchmark.display_name if benchmark else canonical


def benchmark_slug(value: str) -> str:
    canonical = canonical_benchmark_id(value)
    return BENCHMARK_BY_ID[canonical].slug


def benchmark_sort_key(value: str) -> tuple[int, str]:
    canonical = canonical_benchmark_id(value, allow_unknown=True)
    order = {benchmark.id: index for index, benchmark in enumerate((*BENCHMARKS, *TEXT_SEARCH_BENCHMARKS))}
    return order.get(canonical, len(order)), canonical


def ordered_benchmark_ids(values: Iterable[str]) -> list[str]:
    canonical = {
        canonical_benchmark_id(value, allow_unknown=True) for value in values
    }
    return sorted(canonical, key=benchmark_sort_key)


def benchmark_catalog() -> list[dict[str, object]]:
    pass
    catalog = []
    for benchmark in (*BENCHMARKS, *TEXT_SEARCH_BENCHMARKS):
        item = asdict(benchmark)
        item["aliases"] = list(benchmark.aliases)
        catalog.append(item)
    return catalog
