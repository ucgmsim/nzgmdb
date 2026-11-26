"""
Requests to the SeismicNow API.
"""

import requests

from nzgmdb.management import slack

SEISMIC_NOW_URL = "https://quakecoresoft.canterbury.ac.nz/seismicnow/api/"
# The update re-reads the event's flatfiles, so allow it some time
SEISMIC_NOW_TIMEOUT_S = 120


def format_response_error(response: requests.Response) -> str:
    """
    Format a failed SeismicNow response for a Slack message.

    Parameters
    ----------
    response : requests.Response
        The failed response.

    Returns
    -------
    str
        Status code and the start of the response text.
    """
    return f"status {response.status_code}: `{response.text[: slack.MAX_ERROR_CHARS]}`"


def update_event_cmt(event_id: str, thread: slack.EventThread) -> bool:
    """
    Ask SeismicNow to update an event, applying its new CMT solution and runs.

    SeismicNow replies in the event's Slack thread when the solution is applied,
    so only failures are reported here.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID.
    thread : slack.EventThread
        Slack thread of the event, for failure replies.

    Returns
    -------
    bool
        True if SeismicNow updated the event.
    """
    try:
        response = requests.post(
            f"{SEISMIC_NOW_URL}earthquakes/update",
            json={"earthquake_id": event_id},
            timeout=SEISMIC_NOW_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        error = slack.format_error(exc)
    else:
        if response.status_code == 200:
            print(f"[seismic_now] updated {event_id}: {response.text}")
            return True
        error = format_response_error(response)

    print(f"[seismic_now] failed to update {event_id}: {error}")
    thread.reply(
        f"❌ Failed to add the CMT solution to SeismicNow ({error}), SW team investigate"
    )
    return False
