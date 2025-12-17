"""
Hardcoded prompt suite used for probing fidelity and drift.
"""

PROMPTS = {
    "merge_sort": """You are a Python tutor. Write a clear and concise iterative merge sort implementation in Python. Only output the function code.""",
    "bst": """Design a binary search tree API in Python with insert, search, delete, and an inorder traversal generator. Return only code.""",
    "sudoku": """Solve this Sudoku (empty cells are 0):
5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9
Respond with the completed grid only.""",
}

PROMPT_ORDER = ["merge_sort", "bst", "sudoku"]


def get_prompt(name: str) -> str:
    return PROMPTS[name]


def all_prompts():
    for name in PROMPT_ORDER:
        yield name, PROMPTS[name]
