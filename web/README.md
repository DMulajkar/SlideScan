# SlideScan web jobs

The page is `dhruvmulajkar.me/slidescan/` (repo `final-portfolio`, file `slidescan/index.html`). It sends a walkaround video to this server on the Azure VM, which runs the same pipeline as the PC runs and serves the results.

- `server.py`: stdlib HTTP API, one job at a time, an access code for uploads. The API is listed in its docstring.
- `run_job.sh`: one job. It tone-maps HDR to SDR, then runs frames, COLMAP, heights and score with the Command Reference settings, then `replay.py --red-mm 5.5` for `reconstruction.mp4` (bumps at the score's cut in red; if the video fails, the results still stand). It deletes the video and frames when done.
- `test_server.py`: self-test with a fake pipeline. Run `python web/test_server.py`, which prints `ok`.
- `slidescan.service`: the systemd unit. `Caddyfile`: HTTPS in front of the server.

## On the VM (done 2026-11-12)
```
sudo cp web/slidescan.service /etc/systemd/system/ && sudo systemctl enable --now slidescan
# ~/slidescan.env (mode 600): SLIDESCAN_TOKEN=<access code>; a copy is on the PC at ~/.ssh/slidescan_access_code.txt
sudo apt install caddy && sudo cp web/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
```
- HTTPS needs inbound TCP 443 open in the VM's network security group. The hostname is `slidescan-api.northcentralus.cloudapp.azure.com`, a DNS label on the VM's static IP.
- **Start and stop (2026-11-13):**
  - `SLIDESCAN_IDLE_MIN=20` in `~/slidescan.env` makes the server deallocate its own VM after 20 idle minutes, using the VM's system-assigned managed identity.
  - The page's Start button calls `final-portfolio/api/slidescan-start.js`, a Vercel function that uses the service principal `slidescan-starter`. Its secret is on the PC at `~/.ssh/slidescan_starter.json` and in Vercel's Production variables, and expires 2027-11-13.
  - Both identities hold only the custom role "SlideScan VM Power" (start, deallocate and read), assigned on this VM.
- Jobs live in `~/slidescan-jobs/<id>/` (`state.json`, `job.log`, `out/`). A job that was waiting or running when the server restarted is marked failed.

## Day to day
- **Start the processor:** the page starts it on upload, or with its Start button (access code needed). From a terminal: `az vm start -g slidescan-rg -n slidescan-vm`. The server starts on boot.
- **Stop it:** it switches itself off after 20 idle minutes. To stop it now: `az vm deallocate -g slidescan-rg -n slidescan-vm`. Auto-shutdown also deallocates it daily at 04:00 UTC.
- **Run a command without SSH** (for example if the home IP changed): `az vm run-command invoke -g slidescan-rg -n slidescan-vm --command-id RunShellScript --scripts "..."`, which runs as root.
- **Logs:** `journalctl -u slidescan -f` on the VM.
- **Local preview of the page:** serve `final-portfolio` on `localhost:8765`, then run `ssh -L 8080:127.0.0.1:8080 azureuser@<vm>`. On localhost the page talks to `http://localhost:8080`.
- **Uploading with curl from Windows:** add `--http1.1`. Over HTTP/2, Windows curl (schannel) dropped a 196 MB upload after about 6 s with `SEC_E_MESSAGE_ALTERED`; HTTP/1.1 took 65 s.
- **Editing these files on Windows:** keep LF line endings. A `run_job.sh` with CRLF fails at once with `set: pipefail\r: invalid option name`.

## Limits
- Test board only: `board.json` gives the scale and the reference plane.
- Heights are raw, with no offset correction. The tone-mapped tex1 run read +2.98 mm.
- The focal length is fixed at 26 mm (iPhone main camera).
- The PQ (HDR10) tone-map path is untested; only iPhone HLG has been tried.
- Results can be read only while the VM is on. The Start button brings it back in about two minutes.
