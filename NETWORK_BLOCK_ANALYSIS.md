# Network Block Investigation Notes

Date analyzed: 2026-05-06 (UTC)

## What we can determine from this repository

- There are no runtime web proxy/security logs committed that show the exact time of a corporate network block event.
- `serve-log.txt` only contains build-related output (UTF-16 encoded), not a firewall/proxy block entry.
- `nginx.conf` defines the `access_log` format with request timestamps (`$time_local`), but does not include stored log events in the repo.

## Likely block-trigger indicators present in site code/content

1. **Stealth-mode naming and UI behavior**
   - File names and UI labels explicitly use `stealth` (e.g., `templates/stealth-mode.html`, `data-stealth`, `Stealth On/Off`).
   - Some enterprise filters flag “stealth” terminology as obfuscation/evasion.

2. **External tunnel domain in browser WebSocket**
   - The frontend attempts a WebSocket connection to:
     - `wss://precut-gigabyte-ambulance.ngrok-free.dev/frontend-transcript`
   - Many corporate firewalls block or heavily score `ngrok`/tunnel domains as data exfiltration risk.

3. **Medical/cannabis terminology in consult types**
   - The UI includes text like `Medicinal cannabis / CBD / THC consult`.
   - Certain enterprise content filters can flag drug-related keywords depending on policy.

## Time indicators available

- Potential timestamp source configured: NGINX access log line uses `$time_local`.
- Exact block time cannot be recovered from committed files because no actual access/error log entries are present in this repository.

## Recommended next checks (outside this repo)

1. Pull the exact corporate block event from your network security stack (Zscaler, Umbrella, Palo Alto, Netskope, etc.) using your site domain and **UTC + local timezone**.
2. Review server-side logs around the same period:
   - `/var/log/nginx/access.log`
   - `/var/log/nginx/error.log`
   - hosting platform request logs.
3. Temporarily disable/rename stealth-related UI labels and remove ngrok endpoint from production frontend to test whether block status clears.

## Direct answer to your question

Based on this repo inspection, **yes, it is plausible your work internet/network blocked the site**.
The strongest in-repo indicators are:
- Stealth-mode wording/UX markers.
- A frontend connection to an `ngrok-free.dev` tunnel endpoint.
- Policy-sensitive medical/drug terminology in visible UI content.

However, this repository alone cannot prove the exact block reason or exact block timestamp; confirmation must come from your corporate web filter/security event logs.
