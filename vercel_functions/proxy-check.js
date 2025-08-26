export const config = { runtime: 'edge' };

// Edge function to hide the demo API key. Set environment variables in Vercel:
// DEMO_API_KEY = test_free_key
// API_BASE_URL = https://api-spam-ui90.onrender.com

export default async function handler(req) {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405, headers: { 'content-type': 'application/json' } });
  }
  const { text } = await req.json().catch(() => ({}));
  if (!text || typeof text !== 'string' || text.trim().length === 0) {
    return new Response(JSON.stringify({ error: 'Missing text' }), { status: 400, headers: { 'content-type': 'application/json' } });
  }
  const apiKey = process.env.DEMO_API_KEY;
  const base = process.env.API_BASE_URL || 'https://api-spam-ui90.onrender.com';
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'Demo key not configured' }), { status: 500, headers: { 'content-type': 'application/json' } });
  }
  try {
    const url = new URL(base);
    url.pathname = (url.pathname.replace(/\/$/, '') + '/check');
    const res = await fetch(url.toString(), {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'X-API-Key': apiKey },
      body: JSON.stringify({ text })
    });
    const bodyText = await res.text();
    return new Response(bodyText, { status: res.status, headers: { 'content-type': 'application/json' } });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), { status: 502, headers: { 'content-type': 'application/json' } });
  }
}


