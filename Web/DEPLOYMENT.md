# Deployment Notes

Docker is no longer the recommended deployment path for this project.
Use `systemd --user` for the backend and the existing Cloudflare tunnel service for public traffic.

Target hostnames:

- `lilkremxxx.strokeg.pp.ua`
- `strokeg.lilkremxxx.pp.ua`

## Local prerequisites

```bash
cd /home/nhatbang/Samsung_NIC/Healthcare-Stroke-prediction/Web
cp be/.env.example be/.env
python3 -m venv be/.venv
be/.venv/bin/pip install -r be/requirements.txt
cd fe
/home/nhatbang/.nvm/versions/node/v25.7.0/bin/npm install
/home/nhatbang/.nvm/versions/node/v25.7.0/bin/npm run build
```

## systemd backend

On this machine, the live backend is already managed by `verbaa-backend.service` and the live tunnel is `cloudflared.service`.
Those services already point at this checkout, so the fastest deploy path is to restart them after rebuilding the frontend/backend.

If you want a dedicated StrokeGuard unit, install the template backend unit into `~/.config/systemd/user/`:

```bash
cp /home/nhatbang/Samsung_NIC/Healthcare-Stroke-prediction/Web/deploy/systemd/strokeguard-backend.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now strokeguard-backend.service
```

The backend listens on `127.0.0.1:9001`.

## Cloudflare tunnel

Reuse the existing user-level `cloudflared` service and keep the public URLs in `be/.env` aligned with the hostname that the tunnel exposes.
On this machine, `verbaa.pp.ua` is the known working public hostname for the tunnel.

## Fast path

```bash
cd /home/nhatbang/Samsung_NIC/Healthcare-Stroke-prediction/Web
docker compose down
systemctl --user restart verbaa-backend.service cloudflared.service
```

## Notes

- `Web/fe/dist` is served by the FastAPI app, so you only need to rebuild the frontend when UI files change.
- The old Docker files remain in the repo, but they are no longer the deployment path.
