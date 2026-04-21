import praw
import os
import json # Added for pretty printing test output

def fetch_recent_posts(subreddit_name, limit=1):
    """
    Fetches recent posts from a specified subreddit.
    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables to be set.
    """
    # Ensure user_agent is unique and descriptive for your application
    user_agent = os.getenv("REDDIT_USER_AGENT", "python:my-reddit-fetcher:v1.0 (by /u/YOUR_REDDIT_USERNAME)")
    
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=user_agent,
    )
    
    if not reddit.client_id or not reddit.client_secret:
        raise ValueError("Reddit client ID or secret not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    print(f"Fetching {limit} new posts from r/{subreddit_name}...")
    try:
        for submission in subreddit.new(limit=limit):
            posts.append({
                "title": submission.title,
                "url": submission.url,
                "score": submission.score,
                "id": submission.id,
                "created_utc": submission.created_utc
            })
        print(f"Successfully fetched {len(posts)} posts.")
    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name}: {e}")
        # Depending on the error, PRAW might give more specific exceptions (e.g., praw.exceptions.PRAWException)
    return posts

if __name__ == "__main__":
    # --- Test Reddit Fetch Functionality ---
    print("\n--- Initiating Reddit Data Fetch Test ---")
    
    # It's highly recommended to set these as environment variables.
    # For local testing, you might load them from a .env file.
    # Example: pip install python-dotenv, then dotenv.load_dotenv()

    # Ensure necessary environment variables are set before running
    if not os.getenv("REDDIT_CLIENT_ID") or not os.getenv("REDDIT_CLIENT_SECRET"):
        print("WARNING: REDDIT_CLIENT_ID and/or REDDIT_CLIENT_SECRET environment variables are not set.")
        print("Please set them to successfully fetch data from Reddit.")
        print("Example: export REDDIT_CLIENT_ID='your_client_id'\n         export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("Skipping fetch test due to missing credentials.")
    else:
        try:
            # Define parameters for the test fetch
            test_subreddit = "Python" # Using a general subreddit for broader accessibility
            fetch_limit = 3 # Fetch a small number of posts for a quick test

            # Call the fetch function
            fetched_data = fetch_recent_posts(test_subreddit, limit=fetch_limit)

            if fetched_data:
                print(f"\nFetched {len(fetched_data)} post(s) from r/{test_subreddit}:\n")
                # Pretty print the results for readability
                print(json.dumps(fetched_data, indent=2))
            else:
                print(f"No posts were fetched from r/{test_subreddit}. Check logs for potential errors or a very inactive subreddit.")

        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during the test: {e}")

    print("\n--- Reddit Data Fetch Test Complete ---")
