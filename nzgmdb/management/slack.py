"""
Helpers for posting, replying to and updating messages in Slack.
"""

import datetime
import os
from zoneinfo import ZoneInfo

import requests

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")
SLACK_API_URL = "https://slack.com/api"
SLACK_TIMEOUT_S = 10

NZ_TZ = ZoneInfo("Pacific/Auckland")
GEONET_EVENT_URL = "https://www.geonet.org.nz/earthquake/{event_id}"
SEISMIC_NOW_EVENT_URL = (
    "https://quakecoresoft.canterbury.ac.nz/seismic_now/map?event={event_id}"
)

# Longest error text put in a Slack message
MAX_ERROR_CHARS = 500

# Custom workspace emoji (beachball image) marking CMT messages
CMT_EMOJI = ":cmt_beachball:"

# Parent message status prefixes, kept in sync with SeismicNow's backend/api/slack.py
STATUS_PROCESSING = "⏳ Processing"
STATUS_BASIC = "🟡 Basic processing on SeismicNow"
STATUS_FINAL = "✅ Final processing on SeismicNow"
STATUS_CMT_RUNNING = f"{CMT_EMOJI} CMT running"
STATUS_CMT_DONE = "✅ CMT done"
STATUS_CMT_FAILED = "⚠️ CMT failed"
STATUS_FAILED = "❌ Processing failed"
STATUS_REMOVED = "🗑️ Removed from SeismicNow"


def _post(method: str, data: dict) -> dict:
    """
    Call a Slack Web API method on the configured channel.

    Parameters
    ----------
    method : str
        Slack API method, e.g. "chat.postMessage".
    data : dict
        Request body, the channel is added.

    Returns
    -------
    dict
        The response from the Slack API.

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
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{SLACK_API_URL}/{method}",
        headers=headers,
        json={"channel": SLACK_CHANNEL, **data},
        timeout=SLACK_TIMEOUT_S,
    )
    response_data = response.json()

    if not response_data.get("ok"):
        raise ValueError(f"Error calling {method}: {response_data}")

    return response_data


def send_message_to_slack(message: str) -> dict:
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
    """
    return _post("chat.postMessage", {"text": message})


def reply_to_message_on_slack(thread_ts: str, reply_message: str) -> dict:
    """
    Reply to a message in Slack (threaded reply), prefixed with the current NZ time.

    Parameters
    ----------
    thread_ts : str
        The timestamp of the parent message to reply to.
    reply_message : str
        The reply text.

    Returns
    -------
    dict
        The response JSON containing the message timestamp (ts)
    """
    return _post(
        "chat.postMessage",
        {"text": f"`{format_nz_time()}` {reply_message}", "thread_ts": thread_ts},
    )


def update_message_on_slack(ts: str, message: str) -> dict:
    """
    Replace the text of a message already posted to Slack.

    Parameters
    ----------
    ts : str
        The timestamp of the message to update.
    message : str
        The new message text.

    Returns
    -------
    dict
        The response from the Slack API.
    """
    return _post("chat.update", {"ts": ts, "text": message})


def format_nz_time(time: datetime.datetime | None = None) -> str:
    """
    Format a time in NZ local time.

    Parameters
    ----------
    time : datetime.datetime, optional
        Time to format, naive times are taken as UTC, by default now.

    Returns
    -------
    str
        e.g. "2026-11-02 14:03:12 NZDT".
    """
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    elif time.tzinfo is None:
        time = time.replace(tzinfo=datetime.UTC)
    return time.astimezone(NZ_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_event_line(
    mag: float,
    lat: float,
    lon: float,
    depth: float,
    quality: str | None = None,
) -> str:
    """
    Format the magnitude, depth, location and quality of an event on one line.

    Parameters
    ----------
    mag : float
        Magnitude.
    lat : float
        Latitude.
    lon : float
        Longitude.
    depth : float
        Depth in km.
    quality : str, optional
        GeoNet quality.

    Returns
    -------
    str
        e.g. "M4.12 | Depth 12.30 km | Lat -41.2000, Lon 174.5000 | Quality preliminary".
    """
    line = f"M{mag:.2f} | Depth {depth:.2f} km | Lat {lat:.4f}, Lon {lon:.4f}"
    if quality:
        line += f" | Quality {quality}"
    return line


def format_event_summary(
    event_id: str,
    mag: float,
    lat: float,
    lon: float,
    depth: float,
    quality: str | None = None,
    locality: str | None = None,
    origin_time: datetime.datetime | None = None,
) -> str:
    """
    Format the summary of an event shown in its parent Slack message.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID.
    mag : float
        Magnitude.
    lat : float
        Latitude.
    lon : float
        Longitude.
    depth : float
        Depth in km.
    quality : str, optional
        GeoNet quality.
    locality : str, optional
        GeoNet locality text, e.g. "5 km south of Napier".
    origin_time : datetime.datetime, optional
        Origin time, naive times are taken as UTC.

    Returns
    -------
    str
        Event ID, origin time, locality, event line and GeoNet / SeismicNow links.
    """
    header = f"*Event {event_id}*"
    if origin_time is not None:
        header += f" | Origin {format_nz_time(origin_time)}"
    if locality:
        header += f" | {locality}"
    links = (
        f"<{GEONET_EVENT_URL.format(event_id=event_id)}|GeoNet> · "
        f"<{SEISMIC_NOW_EVENT_URL.format(event_id=event_id)}|SeismicNow>"
    )
    return f"{header}\n{format_event_line(mag, lat, lon, depth, quality)}\n{links}"


def format_error(exc: BaseException) -> str:
    """
    Format an exception for a Slack message, truncated to MAX_ERROR_CHARS.

    Parameters
    ----------
    exc : BaseException
        The exception.

    Returns
    -------
    str
        e.g. "ValueError: bad value".
    """
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


class EventThread:
    """
    Slack thread for one event, a parent message showing the event status and
    summary with timestamped replies under it.

    Slack failures are printed and never raised, so they can't stop the
    pipeline. Before ``start`` (or without a ts), every call does nothing.
    """

    def __init__(self, ts: str | None = None, summary: str = ""):
        """
        Parameters
        ----------
        ts : str, optional
            Timestamp of an existing parent message.
        summary : str, optional
            Event summary shown in the existing parent message.
        """
        self.ts = ts
        self.summary = summary

    def start(self, status: str, summary: str) -> None:
        """
        Post the parent message for the event.

        Parameters
        ----------
        status : str
            Status prefix, one of the STATUS_* constants.
        summary : str
            Event summary from format_event_summary.
        """
        self.summary = summary
        try:
            self.ts = send_message_to_slack(f"{status} {summary}")["ts"]
        except (ValueError, requests.RequestException) as exc:
            print(f"[management.slack] failed to post event message: {exc}")

    def reply(self, message: str) -> None:
        """
        Reply in the event's thread.

        Parameters
        ----------
        message : str
            The reply text.
        """
        if self.ts is None:
            return
        try:
            reply_to_message_on_slack(self.ts, message)
        except (ValueError, requests.RequestException) as exc:
            print(f"[management.slack] failed to reply in thread {self.ts}: {exc}")

    def set_status(self, status: str, summary: str | None = None) -> None:
        """
        Update the parent message with a new status, and optionally a new summary.

        Parameters
        ----------
        status : str
            Status prefix, one of the STATUS_* constants.
        summary : str, optional
            New event summary, by default the current one is kept.
        """
        if self.ts is None:
            return
        if summary is not None:
            self.summary = summary
        try:
            update_message_on_slack(self.ts, f"{status} {self.summary}")
        except (ValueError, requests.RequestException) as exc:
            print(f"[management.slack] failed to update message {self.ts}: {exc}")
