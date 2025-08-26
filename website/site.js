// Shared JS for navigation, animations, and CTAs

(function() {
    function onReady(fn) {
        if (document.readyState !== 'loading') fn();
        else document.addEventListener('DOMContentLoaded', fn);
    }

    onReady(function() {
        // Init i18n if present
        if (window.I18N && typeof I18N.init === 'function') {
            I18N.init();
        }
        const yearEl = document.getElementById('year');
        if (yearEl) yearEl.textContent = String(new Date().getFullYear());

        // Mobile menu
        const menuBtn = document.getElementById('menuBtn');
        const mobileMenu = document.getElementById('mobileMenu');
        if (menuBtn && mobileMenu) {
            menuBtn.addEventListener('click', () => {
                mobileMenu.classList.toggle('hidden');
            });
        }

        // CTA links    
        const rapidUrl = (window.SITE && window.SITE.rapidApiListingUrl) || 'https://rapidapi.com/meilleramaury/api/spam-detector-api';
        ['ctaRapid','ctaRapidHero','ctaRapidMobile'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.href = rapidUrl;
                if (window.I18N) el.setAttribute('data-i18n', 'cta.tryRapid');
            }
        });

        // Pricing dynamic content and plan links (locale-aware)
        if (window.SITE && SITE.pricing) {
            const setText = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = String(text); };
            const linkTo = (id, href) => { const el = document.getElementById(id); if (el && href) el.href = href; };
            const { pricing, pricingLinks } = SITE;
            const formatPrice = (amount, currency) => new Intl.NumberFormat((window.I18N? I18N.current : 'en'), { style: 'currency', currency }).format(amount) + (I18N && I18N.current === 'fr' ? '/mois' : '/month');

            setText('price-free', formatPrice(pricing.free.price, pricing.free.currency));
            setText('quota-free', pricing.free.quotaMonthly);
            setText('rpm-free', pricing.free.rpm);
            setText('price-pro5k', formatPrice(pricing.pro5k.price, pricing.pro5k.currency));
            setText('quota-pro5k', pricing.pro5k.quotaMonthly);
            setText('rpm-pro5k', pricing.pro5k.rpm);
            setText('price-pro15k', formatPrice(pricing.pro15k.price, pricing.pro15k.currency));
            setText('quota-pro15k', pricing.pro15k.quotaMonthly);
            setText('rpm-pro15k', pricing.pro15k.rpm);
            setText('price-pro50k', formatPrice(pricing.pro50k.price, pricing.pro50k.currency));
            setText('quota-pro50k', pricing.pro50k.quotaMonthly);
            setText('rpm-pro50k', pricing.pro50k.rpm);
            if (pricingLinks) {
                linkTo('buy-free', pricingLinks.free);
                linkTo('buy-pro5k', pricingLinks.pro5k);
                linkTo('buy-pro15k', pricingLinks.pro15k);
                linkTo('buy-pro50k', pricingLinks.pro50k);
            }
        }

        // Language selector
        const langSel = document.getElementById('langSelect');
        if (langSel && window.I18N) {
            langSel.value = I18N.current;
            langSel.addEventListener('change', (e) => {
                I18N.set(e.target.value);
                // re-render localized prices
                if (window.SITE && SITE.pricing) {
                    const { pricing } = SITE;
                    const formatPrice = (amount, currency) => new Intl.NumberFormat(I18N.current, { style: 'currency', currency }).format(amount) + (I18N.current === 'fr' ? '/mois' : '/month');
                    document.getElementById('price-free') && (document.getElementById('price-free').textContent = formatPrice(pricing.free.price, pricing.free.currency));
                    document.getElementById('price-pro5k') && (document.getElementById('price-pro5k').textContent = formatPrice(pricing.pro5k.price, pricing.pro5k.currency));
                    document.getElementById('price-pro15k') && (document.getElementById('price-pro15k').textContent = formatPrice(pricing.pro15k.price, pricing.pro15k.currency));
                    document.getElementById('price-pro50k') && (document.getElementById('price-pro50k').textContent = formatPrice(pricing.pro50k.price, pricing.pro50k.currency));
                }
            });
        }

        // Reveal animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('reveal-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
    });
})();


