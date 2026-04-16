from enum import Enum

class Category(str, Enum):
    EXACT_VALUE = "EXACT_VALUE"
    EXPRESSION = "EXPRESSION"
    PROOF = "PROOF"
    COMPLEX = "COMPLEX"