"""
Helpers for posting and replying to messages in Slack.
"""

import os

import requests

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")


def send_message_to_slack(message: str):
    """
    Send a message to a Slack channel.

    Parameters
    ----------
    message : str
        The message to send.

    Returns
    -------
    dict
        The response from the Slack API, containing the message timestamp.

    Raises
    ------
    ValueError:
        If SLACK_CHANNEL or SLACK_BOT_TOKEN is not set in the environment.
        Or if the response from Slack is not successful.
    """
    if not SLACK_CHANNEL:
        raise ValueError(
            "No slack channel provided from the environment var SLACK_CHANNEL"
        )
    if not SLACK_BOT_TOKEN:
        raise ValueError(
            "No slack bot token provided from the environment var SLACK_BOT_TOKEN"
        )
    url = "https://slack.com/api/chat.postMessage"
    data = {
        "channel": SLACK_CHANNEL,
        "text": message,
    }
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    if not response_data.get("ok"):
        raise ValueError(f"Error sending message: {response_data}")

    return response_data


def reply_to_message_on_slack(thread_ts: str, reply_message: str):
    """
    Reply to a message in Slack (threaded reply).

    Parameters
    ----------
    thread_ts : str
        The timestamp of the message to reply to.
    reply_message : str
        The reply text.

    Returns
    -------
    dict
        The response JSON containing the message timestamp (ts)

    Raises
    ------
    ValueError
        If SLACK_CHANNEL or SLACK_BOT_TOKEN is not set in the environment.
        Or if the response from Slack is not successful.
    """
    if not SLACK_CHANNEL:
        raise ValueError(
            "No slack channel provided from the environment var SLACK_CHANNEL"
        )
    if not SLACK_BOT_TOKEN:
        raise ValueError(
            "No slack bot token provided from the environment var SLACK_BOT_TOKEN"
        )

    url = "https://slack.com/api/chat.postMessage"
    data = {
        "channel": SLACK_CHANNEL,
        "text": reply_message,
        "thread_ts": thread_ts,  # This ensures it's a threaded reply
    }
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    if not response_data.get("ok"):
        raise ValueError(f"Error sending reply: {response_data}")

    return response_data
