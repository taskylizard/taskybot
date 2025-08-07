import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


from dateutil import parser as dateutil_parser

# Configuration
MAX_SAMPLES_TO_SCRAPE = 1000
LOCAL_DATA_BASE_PATH = "Messages"
TARGET_USER_ID = "352837489125228546"

# Channel filtering - specify which channels to include/exclude
INCLUDED_CHANNELS = [
    "956006108164673538",  # FMHY: general
    # "1288962477542866944",  # taskyland: chatroom
    # "974585038685474826",  # FMHY: fm-dev
]

EXCLUDED_CHANNELS = [
    # Add channel names or IDs you want to exclude
    # Examples:
    # "spam",
    # "bot-commands",
    # "unknown",  # Skip channels with no name
]
EXCLUDED_KEYWORDS = {"codenames", "red", "blue", "spymaster"}


def should_include_channel(channel_info: dict) -> bool:
    """Check if a channel should be included based on filtering rules."""
    channel_name = channel_info.get("name", "unknown")
    channel_id = str(channel_info.get("id", ""))

    # If we have an inclusion list, only include channels in that list
    if INCLUDED_CHANNELS:
        return channel_name in INCLUDED_CHANNELS or channel_id in INCLUDED_CHANNELS

    # Otherwise, exclude channels in the exclusion list
    if EXCLUDED_CHANNELS:
        return not (
            channel_name in EXCLUDED_CHANNELS or channel_id in EXCLUDED_CHANNELS
        )

    # If no filters specified, include all channels
    return True


def load_messages_from_local_data(data_path: str, target_user_id: str) -> list[dict]:
    """Load messages from local Discord data export."""
    messages = []
    # Handle both relative and absolute paths
    if not Path(data_path).is_absolute():
        # Try relative to current working directory first
        data_base_path = Path.cwd() / data_path
        if not data_base_path.exists():
            # Try relative to the parent directory (common case)
            data_base_path = Path.cwd().parent / data_path
    else:
        data_base_path = Path(data_path)

    print(f"Looking for data at: {data_base_path.absolute()}")
    if not data_base_path.exists():
        print(f"Data path {data_base_path.absolute()} does not exist")
        return messages

    # Track channel statistics
    channels_processed = 0
    channels_skipped = 0
    total_channels = 0

    # Find all channel directories
    for channel_dir in data_base_path.iterdir():
        if not channel_dir.is_dir():
            continue

        total_channels += 1
        messages_file = channel_dir / "messages.json"
        channel_file = channel_dir / "channel.json"

        if not messages_file.exists() or not channel_file.exists():
            channels_skipped += 1
            continue

        try:
            # Load channel info first to check if we should include it
            with open(channel_file, "r", encoding="utf-8") as f:
                channel_info = json.load(f)

            # Apply channel filtering
            if not should_include_channel(channel_info):
                channels_skipped += 1
                channel_name = channel_info.get("name", "unknown")
                continue

            channels_processed += 1

            # Load messages
            with open(messages_file, "r", encoding="utf-8") as f:
                channel_messages = json.load(f)

            channel_name = channel_info.get("name", "unknown")
            print(
                f"Loaded {len(channel_messages)} messages from channel '{channel_name}' (ID: {channel_info.get('id')})"
            )

            # Convert to standardized format
            for msg in channel_messages:
                # Parse timestamp
                timestamp_str = msg.get("Timestamp", "")
                try:
                    timestamp = dateutil_parser.parse(timestamp_str)
                except Exception:
                    timestamp = datetime.now(timezone.utc)

                standardized_msg = {
                    "id": int(msg.get("ID", 0)),
                    "content": msg.get("Contents", ""),
                    "author_id": target_user_id,  # Assume all messages are from target user in personal export
                    "author_name": "You",  # Placeholder since export doesn't contain author info
                    "timestamp": timestamp.isoformat(),
                    "created_at": timestamp,
                    "bot": False,
                    "channel_id": channel_info.get("id"),
                    "channel_name": channel_info.get("name", "unknown"),
                    "attachments": msg.get("Attachments", ""),
                }
                messages.append(standardized_msg)

        except Exception as e:
            print(f"Error loading messages from {channel_dir}: {e}")
            channels_skipped += 1
            continue

    # Sort messages by timestamp (oldest first)
    messages.sort(key=lambda m: m["created_at"])

    # Print channel statistics
    print("=" * 60)
    print("📊 CHANNEL STATISTICS")
    print("=" * 60)
    print(f"Total channel directories found: {total_channels}")
    print(f"Channels processed: {channels_processed}")
    print(f"Channels skipped: {channels_skipped}")
    print(f"Total messages loaded: {len(messages)}")
    print("=" * 60)

    return messages


def process_conversations_from_local_data(
    messages: list[dict],
    target_user_id: str,
    min_message_length: int,
    cutoff_days: int,
    limit: int,
) -> list[dict]:
    """Process conversations from locally loaded messages."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
    conversations = []

    # Filter messages by cutoff date and target user
    target_messages = []
    for msg in messages:
        # Ensure both timestamps are timezone-aware for comparison
        msg_time = msg["created_at"]
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)
        content = msg["content"].strip()
        is_mention_only = (
            content.startswith("<@") and content.endswith(">") and " " not in content
        )
        is_link_only = bool(re.fullmatch(r"https?://\S+", content))
        is_not_good_link_only = bool(
            re.fullmatch(
                r"(https?://)?(www\.)?(wotaku\.wiki|tenor\.com)(/\S*)?", content
            )
        )
        contains_excluded_keywords = any(
            kw in content.lower() for kw in EXCLUDED_KEYWORDS
        )
        if (
            msg_time > cutoff
            and len(content) >= min_message_length
            and not is_mention_only
            and not is_link_only
            and not is_not_good_link_only
            and not contains_excluded_keywords
        ):
            target_messages.append(msg)

    print(f"Found {len(target_messages)} messages from target user after cutoff date")

    # Create simple conversations - each target message becomes a conversation
    for msg in target_messages[:limit]:
        # Create a conversation with a user prompt and assistant response
        conversation_data = [
            {
                "role": "user",  # Add a generic user message to prompt the assistant
                "content": "Continue the conversation in your typical style.",
            },
            {
                "role": "assistant",  # The target user's message becomes the assistant response
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "message_id": str(msg["id"]),
                "channel": msg["channel_name"],
            },
        ]

        conversations.append({"messages": conversation_data})

        if len(conversations) >= limit:
            break

    print(f"Created {len(conversations)} conversations from local data")
    return conversations


def scrape(
    user_id: Optional[str] = None,
    data_path: Optional[str] = None,
    min_message_length: int = 100,
    cutoff_days: int = 365,
    included_channels: Optional[list[str]] = None,
    excluded_channels: Optional[list[str]] = None,
):
    """

    Args:
        user_id: Discord user ID to target
        data_path: Path to Messages directory
        min_message_length: Minimum message length to include
        cutoff_days: Days in the past to fetch from
        included_channels: List of channel names/IDs to include (None = use global config)
        excluded_channels: List of channel names/IDs to exclude (None = use global config)
    """
    # Update global channel filters if provided
    global INCLUDED_CHANNELS, EXCLUDED_CHANNELS
    if included_channels is not None:
        INCLUDED_CHANNELS = included_channels
    if excluded_channels is not None:
        EXCLUDED_CHANNELS = excluded_channels

    user_id = user_id or TARGET_USER_ID
    data_path = data_path or LOCAL_DATA_BASE_PATH

    if isinstance(user_id, str) and user_id == "YOUR_USER_ID_HERE":
        raise ValueError(
            "Please set TARGET_USER_ID in scrape.py or pass user_id parameter"
        )

    print(f"Beginning local data scrape for user {user_id}...")
    print(f"Data path: {data_path}")
    print(f"Target: {MAX_SAMPLES_TO_SCRAPE} conversations")
    print(f"Min message length: {min_message_length}")

    # Print channel filtering info
    if INCLUDED_CHANNELS:
        print(f"Including only channels: {', '.join(INCLUDED_CHANNELS)}")
    if EXCLUDED_CHANNELS:
        print(f"Excluding channels: {', '.join(EXCLUDED_CHANNELS)}")

    print("=" * 60)

    overall_start_time = time.time()

    # Load all messages from local data
    messages = load_messages_from_local_data(data_path, str(user_id))

    if not messages:
        print("No messages found in local data")
        return 0

    # Process conversations from the loaded messages
    conversations = process_conversations_from_local_data(
        messages=messages,
        target_user_id=str(user_id),
        min_message_length=min_message_length,
        cutoff_days=cutoff_days,
        limit=MAX_SAMPLES_TO_SCRAPE,
    )

    overall_elapsed = time.time() - overall_start_time

    # Save locally for testing
    output_path = Path(f"corpus/{user_id}.json")

    # Limit to maximum samples
    conversations = conversations[:MAX_SAMPLES_TO_SCRAPE]

    with open(output_path, "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"📁 Saved to: {output_path}")
    samples_length = len(conversations)

    # Final summary with comprehensive statistics
    print("🎉 LOCAL DATA SCRAPING COMPLETED")
    print(f"📊 User ID: {user_id}")
    print(f"📈 Conversations found: {samples_length:,}")
    print(f"📁 Saved to: {output_path}")
    print(f"⏱️  Total time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} minutes)")
    if samples_length > 0:
        print(
            f"⚡ Average time per conversation: {overall_elapsed / samples_length:.2f}s"
        )

    return samples_length


result = scrape(
    user_id="352837489125228546",
    data_path="Messages",
    min_message_length=5,
    cutoff_days=365,
)
print(f"\nProcessed {result} conversations from all channels")
