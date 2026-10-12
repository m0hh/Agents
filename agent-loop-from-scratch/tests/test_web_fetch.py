"""
Tests for the fetch_url tool, with a FAKE network and FAKE DNS.

No real web requests happen here:
  * `httpx2.MockTransport(handler)` answers HTTP requests from a dict of URL → response.
  * `fake_dns` answers hostname lookups from a dict of host → IP addresses.
That lets us test the dangerous cases (private IPs, redirect tricks) safely.
"""

import httpx2
import pytest

from agent_loop.tools import ToolError
from agent_loop.tools.web_fetch import make_fetch_url_tool

# Hostname → IP addresses, as our fake DNS server sees the world.
DNS = {
    "example.com": ["93.184.215.14"],       # a public address
    "evil.example": ["93.184.215.99"],      # public too... but it redirects somewhere bad
    "intranet.corp": ["10.0.0.5"],          # a private address
}

PAGE = b"""<html><head><title>Example Page</title><style>body {color: red}</style></head>
<body><h1>Hello</h1><script>alert('tracking')</script><p>Readable   paragraph.</p></body></html>"""


def fake_dns(host: str) -> list[str]:
    if host not in DNS:
        raise OSError("name not found")  # what real DNS failures raise
    return DNS[host]


def make_tool(routes: dict[str, httpx2.Response], **kwargs):
    """A fetch_url tool whose 'internet' is just `routes` (URL → canned response)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if url not in routes:
            return httpx2.Response(404)
        return routes[url]

    return make_fetch_url_tool(transport=httpx2.MockTransport(handler), resolve=fake_dns, **kwargs)


def html(body: bytes) -> httpx2.Response:
    return httpx2.Response(200, content=body, headers={"content-type": "text/html; charset=utf-8"})


def test_html_is_converted_to_clean_labelled_text() -> None:
    tool = make_tool({"https://example.com/": html(PAGE)})

    result = tool.execute({"url": "https://example.com/"})

    assert "Title: Example Page" in result
    assert "Hello" in result
    assert "Readable paragraph." in result  # extra spaces collapsed
    assert "alert" not in result            # <script> removed
    assert "color: red" not in result       # <style> removed
    assert "<<<BEGIN UNTRUSTED WEB CONTENT>>>" in result


def test_long_pages_are_truncated() -> None:
    long_page = b"<html><body><p>" + b"word " * 5000 + b"</p></body></html>"
    tool = make_tool({"https://example.com/": html(long_page)}, max_output_chars=100)

    result = tool.execute({"url": "https://example.com/"})

    assert "[Truncated: showing the first 100 of" in result


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("file:///etc/passwd", "Only http"),
        ("ftp://example.com/", "Only http"),
        ("http://127.0.0.1:8000/admin", "Blocked"),                       # your own machine
        ("http://169.254.169.254/latest/meta-data/", "Blocked"),           # cloud credentials endpoint
        ("http://[::1]/", "Blocked"),                                      # IPv6 localhost
        ("http://[::ffff:127.0.0.1]/", "Blocked"),                         # IPv4 hidden inside IPv6
        ("http://intranet.corp/", "Blocked"),                              # hostname → private IP
        ("http://no-such-host.invalid/", "Could not resolve"),
    ],
)
def test_dangerous_or_invalid_urls_are_refused(url: str, message: str) -> None:
    tool = make_tool({})
    with pytest.raises(ToolError, match=message):
        tool.execute({"url": url})


def test_a_redirect_to_a_private_address_is_blocked() -> None:
    # The classic SSRF trick: a PUBLIC page that redirects to a PRIVATE one.
    tool = make_tool(
        {"https://evil.example/": httpx2.Response(302, headers={"location": "http://intranet.corp/secrets"})}
    )
    with pytest.raises(ToolError, match="Blocked"):
        tool.execute({"url": "https://evil.example/"})


def test_safe_redirects_are_followed() -> None:
    tool = make_tool(
        {
            "https://example.com/old": httpx2.Response(301, headers={"location": "/new"}),
            "https://example.com/new": html(b"<p>Moved here</p>"),
        }
    )
    result = tool.execute({"url": "https://example.com/old"})
    assert "URL: https://example.com/new" in result
    assert "Moved here" in result


def test_redirect_loops_stop() -> None:
    tool = make_tool({"https://example.com/loop": httpx2.Response(302, headers={"location": "/loop"})})
    with pytest.raises(ToolError, match="Too many redirects"):
        tool.execute({"url": "https://example.com/loop"})


def test_http_errors_and_binary_content_are_reported() -> None:
    tool = make_tool(
        {
            "https://example.com/image.png": httpx2.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}),
        }
    )
    with pytest.raises(ToolError, match="HTTP 404"):
        tool.execute({"url": "https://example.com/missing"})
    with pytest.raises(ToolError, match="Unsupported content type"):
        tool.execute({"url": "https://example.com/image.png"})


def test_network_timeouts_become_tool_errors() -> None:
    def slow_handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    tool = make_fetch_url_tool(transport=httpx2.MockTransport(slow_handler), resolve=fake_dns)
    with pytest.raises(ToolError, match="Timed out"):
        tool.execute({"url": "https://example.com/"})


def test_plain_text_and_json_are_returned_as_is() -> None:
    tool = make_tool(
        {"https://example.com/data.json": httpx2.Response(200, json={"answer": 42})}
    )
    result = tool.execute({"url": "https://example.com/data.json"})
    assert '"answer"' in result and "42" in result
