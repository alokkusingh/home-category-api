from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExpenseCategorizationRequest(_message.Message):
    __slots__ = ("head", "comment", "amount")
    HEAD_FIELD_NUMBER: _ClassVar[int]
    COMMENT_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    head: str
    comment: str
    amount: float
    def __init__(self, head: _Optional[str] = ..., comment: _Optional[str] = ..., amount: _Optional[float] = ...) -> None: ...

class ExpenseCategorizationResponse(_message.Message):
    __slots__ = ("category", "confidence", "needs_review")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    NEEDS_REVIEW_FIELD_NUMBER: _ClassVar[int]
    category: str
    confidence: float
    needs_review: bool
    def __init__(self, category: _Optional[str] = ..., confidence: _Optional[float] = ..., needs_review: bool = ...) -> None: ...
