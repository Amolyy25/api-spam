// Global site config for the static website
// Replace placeholder values with your actual RapidAPI details and links.

window.SITE = {
    name: 'Spam Detector API',
    rapidApiListingUrl: 'https://rapidapi.com/meilleramaury/api/spam-detector-api', // Optional: set to your listing URL
    rapidApiHostDefault: 'api-spam-ui90.onrender.com',   // Optional: RapidAPI host if you use it
    baseUrlDefault: 'https://api-spam-ui90.onrender.com',
    githubUrl: 'https://github.com/meilleramaury/spam-detector-api',
    pricing: {
        // Prices in EUR; formatting is locale-aware in site.js
        free:   { price: 0,  currency: 'EUR', quotaMonthly: 100,   rpm: 5 },
        pro5k:  { price: 10, currency: 'EUR', quotaMonthly: 5000,  rpm: 120 },
        pro15k: { price: 25, currency: 'EUR', quotaMonthly: 15000, rpm: 180 },
        pro50k: { price: 50, currency: 'EUR', quotaMonthly: 50000, rpm: 240 }
    },
    // Optional direct links to plan checkout on RapidAPI (set these to your plan URLs)
    pricingLinks: {
        free: '',
        pro5k: '',
        pro15k: '',
        pro50k: ''
    },
    // Endpoint exposed by Vercel Edge Function to hide the demo key server-side
    proxyEndpoint: '/proxy-check'
};


