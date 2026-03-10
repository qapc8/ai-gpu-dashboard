export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const apiKey = process.env.CHAT_API_KEY;
  const apiUrl = process.env.CHAT_API_URL || 'https://api.hyperfusion.io/v1/chat/completions';
  const model = process.env.CHAT_MODEL || 'qwen/qwen3-32b';

  if (!apiKey) {
    return res.status(503).json({ error: 'Chat API key not configured on server.' });
  }

  const { messages, max_tokens } = req.body || {};

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'No messages provided.' });
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
      return res.status(upstream.status).json({ error: `Upstream API error: ${JSON.stringify(data).substring(0, 200)}` });
    }

    return res.status(200).json(data);
  } catch (e) {
    return res.status(502).json({ error: `Connection to AI service failed: ${e.message}` });
  }
}
