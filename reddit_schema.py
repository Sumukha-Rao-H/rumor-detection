from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


class RedditPost(BaseModel):
    """
    Schema for storing a Reddit submission (post) and its metadata.
    """
    id: str = Field(..., description="Unique ID of the post assigned by Reddit (e.g., 't3_123abc').")
    title: str = Field(..., description="Title of the Reddit post.")
    author: str = Field(..., description="Username of the post author.")
    subreddit: str = Field(..., description="Name of the subreddit where the post was made (e.g., 'MachineLearning').")
    created_utc: float = Field(..., description="UTC Unix timestamp when the post was created.")
    score: int = Field(..., description="Current net upvotes of the post (upvotes - downvotes).")
    num_comments: int = Field(..., description="Total number of top-level comments on the post.")
    body: Optional[str] = Field(
        default=None,
        description="Self-text content of the post. Only present for self-posts (is_self=True)."
    )
    url: Optional[HttpUrl] = Field(
        default=None,
        description="The URL the post links to. Only present for link posts (is_self=False)."
    )
    permalink: str = Field(
        ...,
        description="Relative URL path to the post on Reddit (e.g., '/r/subreddit/comments/id/title/')."
    )
    is_self: bool = Field(..., description="True if the post is a self-post (text post), False otherwise.")
    is_video: bool = Field(..., description="True if the post content is a video, False otherwise.")
    link_flair_text: Optional[str] = Field(
        default=None,
        description="Text of the link flair applied to the post, if any."
    )
    upvote_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of upvotes to total votes (0.0 to 1.0). May not be available for very old posts."
    )
    domain: str = Field(
        ...,
        description="The domain of the linked content (e.g., 'youtube.com'), or 'self.subreddit' for self-posts."
    )
    edited_utc: Optional[float] = Field(
        default=None,
        description="UTC Unix timestamp when the post was last edited, if it has been edited."
    )
    over_18: bool = Field(..., description="True if the post is marked as NSFW (Not Safe For Work).")


class RedditComment(BaseModel):
    """
    Schema for storing a Reddit comment and its metadata.
    """
    id: str = Field(..., description="Unique ID of the comment assigned by Reddit (e.g., 't1_abc123').")
    post_id: str = Field(..., description="The ID of the post this comment belongs to (e.g., 't3_123abc').")
    parent_id: str = Field(
        ...,
        description="The ID of the parent item. Can be a post ('t3_...') or another comment ('t1_...')."
    )
    author: str = Field(..., description="Username of the comment author.")
    subreddit: str = Field(..., description="Name of the subreddit where the comment was made.")
    created_utc: float = Field(..., description="UTC Unix timestamp when the comment was created.")
    score: int = Field(..., description="Current net upvotes of the comment (upvotes - downvotes).")
    body: str = Field(..., description="The full text content of the comment.")
    permalink: str = Field(
        ...,
        description="Relative URL path to the comment on Reddit (e.g., '/r/subreddit/comments/id/title/comment_id/')."
    )
    is_submitter: bool = Field(..., description="True if the comment author is also the original poster of the submission.")
    depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="The depth of the comment in the comment tree, where 0 is a top-level comment."
    )
    edited_utc: Optional[float] = Field(
        default=None,
        description="UTC Unix timestamp when the comment was last edited, if it has been edited."
    )
    distinguished: Optional[str] = Field(
        default=None,
        description="The distinguished status of the comment (e.g., 'moderator', 'admin'), if any."
    )
    stickied: bool = Field(..., description="True if the comment is stickied to the top of the comment section.")
