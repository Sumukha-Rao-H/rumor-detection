from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class RedditUser(BaseModel):
    id: Optional[str] = None # Reddit user IDs are often t2_ prefixed for full_name, but pure ID can be useful
    name: str
    is_mod: Optional[bool] = False
    is_employee: Optional[bool] = False

class Subreddit(BaseModel):
    id: Optional[str] = None # e.g., t5_2sjnq
    name: str = Field(..., description="Subreddit display name, e.g., 'python'")
    display_name_prefixed: str = Field(..., description="Subreddit name with 'r/', e.g., 'r/python'")
    subscribers: Optional[int] = 0
    over18: Optional[bool] = False

class Award(BaseModel):
    id: str
    name: str
    count: int
    icon_url: Optional[HttpUrl] = None

class Comment(BaseModel):
    id: str = Field(..., description="Comment ID, e.g., 'gjdj3kl'")
    full_link: Optional[HttpUrl] = None # Full URL to the comment
    author: RedditUser
    body: str
    score: int
    created_utc: datetime
    parent_id: Optional[str] = Field(None, description="ID of the parent comment or post (t1_ or t3_ prefixed)")
    subreddit: Subreddit
    permalink: str # Relative permalink to the comment
    depth: int = 0
    is_submitter: bool = False
    stickied: bool = False
    distinguished: Optional[str] = None # 'moderator', 'admin', 'special'
    locked: bool = False
    controversiality: int = 0
    total_awards_received: int = 0
    all_awardings: List[Award] = []
    replies: List['Comment'] = [] # Nested comments

class Post(BaseModel):
    id: str = Field(..., description="Post ID, e.g., 't3_kxj6e9'")
    title: str
    author: RedditUser
    score: int
    ups: int
    downs: int
    num_comments: int
    created_utc: datetime
    edited_utc: Optional[datetime] = None
    url: HttpUrl = Field(..., description="URL to the external content or permalink if self-post")
    permalink: str = Field(..., description="Relative permalink to the post on Reddit")
    subreddit: Subreddit
    is_self: bool
    selftext: Optional[str] = None
    link_flair_text: Optional[str] = None
    link_flair_css_class: Optional[str] = None
    num_crossposts: int = 0
    stickied: bool = False
    locked: bool = False
    spoiler: bool = False
    over_18: bool = False
    hidden: bool = False
    archived: bool = False
    gilded: int = 0
    total_awards_received: int = 0
    all_awardings: List[Award] = []
    upvote_ratio: Optional[float] = None
    comments: List[Comment] = [] # Top-level comments, or all if fetched recursively

# Pydantic requires forward references for nested models to be updated
Comment.update_forward_refs()
