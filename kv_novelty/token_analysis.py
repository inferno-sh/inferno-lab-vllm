"""
Token-level analysis for understanding spike composition.

This module provides tools to classify tokens and filter spikes
to separate structural (punctuation) from semantic (content) novelty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis import AnalysisResult


@dataclass
class TokenClassification:
    """Classification of a token."""
    token: str
    position: int
    novelty: float
    category: str  # 'content', 'punctuation', 'whitespace', 'code_structure', 'other'
    subcategory: str  # More specific classification


# Patterns for classification
PUNCTUATION = set('.,;:!?()[]{}\'\"')
CODE_KEYWORDS = {'def', 'class', 'if', 'else', 'for', 'while', 'return', 'import',
                 'from', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or',
                 'True', 'False', 'None', 'self'}
WHITESPACE_CHARS = {' ', '\t', '\n', '\r'}


def classify_token(token: str) -> tuple[str, str]:
    """
    Classify a token into category and subcategory.

    Returns:
        Tuple of (category, subcategory)
    """
    stripped = token.strip()

    # Whitespace
    if not stripped or all(c in WHITESPACE_CHARS for c in token):
        if '\n' in token:
            return 'whitespace', 'newline'
        return 'whitespace', 'space'

    # Punctuation
    if stripped in PUNCTUATION or all(c in PUNCTUATION for c in stripped):
        return 'punctuation', 'symbol'

    # Code structure
    if stripped in CODE_KEYWORDS:
        return 'code_structure', 'keyword'
    if stripped.startswith('__') and stripped.endswith('__'):
        return 'code_structure', 'dunder'
    if re.match(r'^[_a-z][_a-z0-9]*$', stripped, re.IGNORECASE):
        # Could be a variable/function name
        if any(c.isupper() for c in stripped) and any(c.islower() for c in stripped):
            return 'content', 'camelcase_identifier'
        if stripped.startswith('_'):
            return 'code_structure', 'private_identifier'

    # Numbers
    if stripped.isdigit() or re.match(r'^-?\d+\.?\d*$', stripped):
        return 'content', 'number'

    # Content words
    if stripped.isalpha():
        if len(stripped) <= 2:
            return 'content', 'short_word'
        return 'content', 'word'

    if stripped.isalnum():
        return 'content', 'alphanumeric'

    # Mixed
    return 'other', 'mixed'


def analyze_spike_composition(
    result: AnalysisResult,
    novelty: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Analyze the composition of detected spikes.

    Args:
        result: Analysis result with spike information
        novelty: Optional novelty array for threshold analysis

    Returns:
        Dictionary with spike composition statistics
    """
    classifications = []

    for pos, token in zip(result.spike_positions, result.spike_tokens):
        category, subcategory = classify_token(token)

        classifications.append(TokenClassification(
            token=token,
            position=int(pos),
            novelty=0.0,  # Can be filled in if novelty array provided
            category=category,
            subcategory=subcategory,
        ))

    # Count by category
    category_counts = {}
    subcategory_counts = {}

    for clf in classifications:
        category_counts[clf.category] = category_counts.get(clf.category, 0) + 1
        key = f"{clf.category}/{clf.subcategory}"
        subcategory_counts[key] = subcategory_counts.get(key, 0) + 1

    total = len(classifications)

    return {
        'total_spikes': total,
        'category_counts': category_counts,
        'subcategory_counts': subcategory_counts,
        'category_percentages': {k: v/total*100 if total > 0 else 0
                                  for k, v in category_counts.items()},
        'classifications': [
            {
                'token': c.token,
                'position': c.position,
                'category': c.category,
                'subcategory': c.subcategory,
            }
            for c in classifications
        ],
        'semantic_spike_rate': (
            category_counts.get('content', 0) / total if total > 0 else 0
        ),
    }


def filter_spikes_by_category(
    result: AnalysisResult,
    include_categories: list[str] | None = None,
    exclude_categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Filter spikes to include/exclude certain token categories.

    Args:
        result: Analysis result
        include_categories: Categories to include (None = all)
        exclude_categories: Categories to exclude

    Returns:
        Tuple of (filtered_positions, filtered_tokens)
    """
    filtered_positions = []
    filtered_tokens = []

    for pos, token in zip(result.spike_positions, result.spike_tokens):
        category, _ = classify_token(token)

        if include_categories is not None and category not in include_categories:
            continue
        if exclude_categories is not None and category in exclude_categories:
            continue

        filtered_positions.append(pos)
        filtered_tokens.append(token)

    return np.array(filtered_positions), filtered_tokens


def compute_semantic_novelty_stats(
    result: AnalysisResult,
    novelty: np.ndarray,
) -> dict[str, Any]:
    """
    Compute novelty statistics separately for semantic vs non-semantic tokens.

    Args:
        result: Analysis result
        novelty: Full novelty array

    Returns:
        Statistics comparing semantic vs non-semantic novelty
    """
    tokens = result.layer_metrics[list(result.layer_metrics.keys())[0]].tokens
    if tokens is None:
        return {}

    semantic_novelty = []
    non_semantic_novelty = []

    for i, token in enumerate(tokens):
        if i >= len(novelty):
            break
        category, _ = classify_token(token)

        if category == 'content':
            semantic_novelty.append(novelty[i])
        else:
            non_semantic_novelty.append(novelty[i])

    semantic_novelty = np.array(semantic_novelty)
    non_semantic_novelty = np.array(non_semantic_novelty)

    return {
        'semantic': {
            'count': len(semantic_novelty),
            'mean': float(semantic_novelty.mean()) if len(semantic_novelty) > 0 else 0,
            'std': float(semantic_novelty.std()) if len(semantic_novelty) > 0 else 0,
            'max': float(semantic_novelty.max()) if len(semantic_novelty) > 0 else 0,
        },
        'non_semantic': {
            'count': len(non_semantic_novelty),
            'mean': float(non_semantic_novelty.mean()) if len(non_semantic_novelty) > 0 else 0,
            'std': float(non_semantic_novelty.std()) if len(non_semantic_novelty) > 0 else 0,
            'max': float(non_semantic_novelty.max()) if len(non_semantic_novelty) > 0 else 0,
        },
        'difference': {
            'mean_diff': (
                (float(semantic_novelty.mean()) - float(non_semantic_novelty.mean()))
                if len(semantic_novelty) > 0 and len(non_semantic_novelty) > 0
                else 0
            ),
        }
    }


def generate_spike_report(result: AnalysisResult) -> str:
    """Generate a detailed spike composition report."""
    composition = analyze_spike_composition(result)

    lines = []
    lines.append("=" * 60)
    lines.append("SPIKE COMPOSITION ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Total spikes: {composition['total_spikes']}")
    lines.append(f"Semantic spike rate: {composition['semantic_spike_rate']*100:.1f}%")
    lines.append("")

    lines.append("Category breakdown:")
    for cat, count in sorted(composition['category_counts'].items(),
                             key=lambda x: -x[1]):
        pct = composition['category_percentages'][cat]
        lines.append(f"  {cat}: {count} ({pct:.1f}%)")
    lines.append("")

    lines.append("Subcategory breakdown:")
    for subcat, count in sorted(composition['subcategory_counts'].items(),
                                key=lambda x: -x[1]):
        lines.append(f"  {subcat}: {count}")
    lines.append("")

    lines.append("Individual spikes:")
    for clf in composition['classifications']:
        lines.append(f"  Pos {clf['position']:3d}: {repr(clf['token']):15s} -> {clf['category']}/{clf['subcategory']}")

    return "\n".join(lines)
