Spam Detector API – Static Website
=================================

Production-ready, responsive static site for the Spam Detector API hosted on RapidAPI.

Pages
-----

- `index.html`: Landing page with hero, benefits, and examples
- `docs.html`: Endpoints, examples (curl, JS, Python), error codes, pricing
- `playground.html`: Live API tester for `POST /check`
- `contact.html`: Contact form via Formspree and links

Configure
---------

Edit `website/config.js`:

- `rapidApiListingUrl`: your RapidAPI listing link
- `rapidApiHostDefault`: e.g., `spam-detector-api.p.rapidapi.com`
- `baseUrlDefault`: e.g., `https://spam-detector-api.p.rapidapi.com`
- `githubUrl`: optional repository link
- Pricing values are prefilled from your `config.yaml`

Deploy
------

Netlify

1. New site from Git
2. Publish directory: `website`
3. Build command: none (static)

Vercel

1. Import project
2. Framework preset: Other
3. Output directory: `website`
4. Build command: none

GitHub Pages

1. Settings → Pages → Deploy from branch
2. Branch: `main`, Folder: `/website`

SEO
---

- Update canonical URLs in each HTML head
- Ensure `og:image` points to a hosted image (`/website/assets/hero-illustration.svg` works when deployed)

Security Notes
--------------

- Playground requires user to provide their own RapidAPI key; it is never stored on server
- Do not hardcode secrets into the frontend

Development
-----------

Open `website/index.html` in your browser. Tailwind is loaded via CDN for zero-build workflow.


