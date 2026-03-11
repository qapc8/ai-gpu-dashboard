// In-memory rate limiter (per Vercel serverless instance)
const rateMap = new Map();
const RATE_LIMIT = 20;       // max requests per window
const RATE_WINDOW = 60000;   // 1 minute window

function isRateLimited(ip) {
  const now = Date.now();
  const entry = rateMap.get(ip);
  if (!entry || now - entry.start > RATE_WINDOW) {
    rateMap.set(ip, { start: now, count: 1 });
    return false;
  }
  entry.count++;
  return entry.count > RATE_LIMIT;
}

// Allowed origins (add your domains here)
const ALLOWED_ORIGINS = [
  'https://ai-gpu-dashboard.vercel.app',
  'https://hyperfusion.io',
  'http://localhost:3000',
  'http://localhost:8080',
  'http://127.0.0.1:5500',
];

function getCorsOrigin(req) {
  const origin = req.headers.origin || '';
  // Allow same-origin requests (no origin header) and listed origins
  if (!origin) return null;
  if (ALLOWED_ORIGINS.some(o => origin.startsWith(o))) return origin;
  // Allow *.vercel.app preview deployments
  if (/^https:\/\/[\w-]+\.vercel\.app$/.test(origin)) return origin;
  return null;
}

export default async function handler(req, res) {
  // CORS — restrict to known origins
  const corsOrigin = getCorsOrigin(req);
  if (corsOrigin) {
    res.setHeader('Access-Control-Allow-Origin', corsOrigin);
    res.setHeader('Vary', 'Origin');
  }
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Rate limiting
  const clientIp = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.socket?.remoteAddress || 'unknown';
  if (isRateLimited(clientIp)) {
    return res.status(429).json({ error: 'Too many requests. Please wait a moment.' });
  }

  const apiKey = process.env.CHAT_API_KEY;
  const apiUrl = process.env.CHAT_API_URL || 'https://api.hyperfusion.io/v1/chat/completions';
  const model = process.env.CHAT_MODEL || 'qwen/qwen3-32b';

  if (!apiKey) {
    return res.status(503).json({ error: 'Chat service unavailable.' });
  }

  const { messages, max_tokens } = req.body || {};

  // Input validation
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'No messages provided.' });
  }

  if (messages.length > 50) {
    return res.status(400).json({ error: 'Too many messages.' });
  }

  // Validate each message
  for (const msg of messages) {
    if (!msg || typeof msg.role !== 'string' || typeof msg.content !== 'string') {
      return res.status(400).json({ error: 'Invalid message format.' });
    }
    if (!['system', 'user', 'assistant'].includes(msg.role)) {
      return res.status(400).json({ error: 'Invalid message role.' });
    }
    if (msg.content.length > 10000) {
      return res.status(400).json({ error: 'Message too long.' });
    }
  }

  const payload = {
    model,
    messages,
    max_tokens: Math.min(max_tokens || 1024, 2048),
    temperature: 0.7,
  };

  try {
    const upstream = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    const data = await upstream.json();

    if (!upstream.ok) {
      // Don't leak upstream error details to client
      return res.status(502).json({ error: 'AI service returned an error. Please try again.' });
    }

    return res.status(200).json(data);
  } catch (e) {
    // Don't leak internal error details
    return res.status(502).json({ error: 'AI service temporarily unavailable.' });
  }
}
