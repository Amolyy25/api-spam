// Lightweight i18n for static pages (EN/FR)

(function() {
  const dict = {
    en: {
      'nav.docs': 'Docs',
      'nav.playground': 'Playground',
      'nav.support': 'Support',
      'cta.tryRapid': 'Try on RapidAPI',

      'home.hero_title': 'Spam Detector API – Real-time Spam, Phishing, and Scam Detection',
      'home.hero_sub': 'Integrate robust, AI-powered content moderation in minutes. High accuracy, low latency, and built-in heuristics for links, keywords, and anomalies.',
      'home.btn_read_docs': 'Read Docs',
      'home.btn_playground': 'Live Playground',
      'home.why_title': 'Why Choose Us?',
      'home.feature_accuracy_title': 'High Accuracy',
      'home.feature_accuracy_desc': 'Hybrid AI + rules engine with tunable thresholds for reliable detection.',
      'home.feature_fast_title': 'Fast Response',
      'home.feature_fast_desc': 'Low-latency inference with caching and rate limiting per plan.',
      'home.feature_ai_title': 'AI-Powered',
      'home.feature_ai_desc': 'Advanced heuristics for suspicious links, forbidden words, and punctuation patterns.',
      'home.examples_title': 'Real Messages. Real Detection.',
      'home.examples.input': 'Input',
      'home.examples.result': 'Result',

      'pricing.title': 'Simple, transparent pricing',
      'pricing.subtitle': 'Scale from hobby to production. Upgrade anytime on RapidAPI.',
      'pricing.note': 'Prices and limits are managed on RapidAPI. Taxes may apply.',
      'pricing.req_per_month': 'req / month',
      'pricing.req_per_minute': 'req / minute',
      'plan.free': 'Free',
      'plan.start_free': 'Start Free',
      'plan.pro5k': 'Pro 5k',
      'plan.pro15k': 'Pro 15k',
      'plan.pro50k': 'Pro 50k',
      'plan.choose_pro5k': 'Choose Pro 5k',
      'plan.choose_pro15k': 'Choose Pro 15k',
      'plan.choose_pro50k': 'Choose Pro 50k',

      'docs.title': 'API Documentation',
      'docs.endpoints': 'Endpoints',
      'docs.examples': 'Examples',
      'docs.errors': 'Error Codes',
      'docs.pricing': 'Pricing',
      'errors.400': 'Invalid payload. Ensure body contains a non-empty "text" string.',
      'errors.401': 'Missing or invalid API key.',
      'errors.403': 'Key not subscribed to a plan or lacks access.',
      'errors.429': 'Rate limit exceeded. Respect per-minute limits by plan.',
      'errors.500': 'Unexpected error. Retry or contact support.',

      'play.title': 'API Playground',
      'play.key': 'RapidAPI Key',
      'play.host': 'RapidAPI Host',
      'play.base_url': 'Base URL',
      'play.use_proxy': 'Use hidden demo key (secure proxy)',
      'play.use_proxy_yes': 'Yes (recommended)',
      'play.use_proxy_no': 'No (I will provide my own key)',
      'play.text_label': 'Text to Analyze',
      'play.btn_check': 'Check Spam',
      'play.btn_save': 'Save settings',
      'play.btn_clear': 'Clear',
      'play.json_response': 'JSON Response',
      'play.msg.enter_text': 'Please enter text to analyze.',
      'play.msg.sending': 'Sending request...',
      'play.msg.saved': 'Settings saved locally.',
      'play.msg.success': 'Success',
      'play.msg.error': 'Request completed with errors',
      'play.msg.network': 'Network error',

      'contact.title': 'Contact & Support',
      'contact.name': 'Name',
      'contact.email': 'Email',
      'contact.message': 'Message',
      'contact.send': 'Send message',
      'contact.github': 'Repository',
    },
    fr: {
      'nav.docs': 'Docs',
      'nav.playground': 'Playground',
      'nav.support': 'Support',
      'cta.tryRapid': 'Essayer sur RapidAPI',

      'home.hero_title': 'Spam Detector API – Détection en temps réel: spam, phishing et arnaques',
      'home.hero_sub': "Intégrez une modération de contenu IA en quelques minutes. Haute précision, faible latence, et heuristiques intégrées pour liens, mots-clés et anomalies.",
      'home.btn_read_docs': 'Lire la documentation',
      'home.btn_playground': 'Playground en direct',
      'home.why_title': 'Pourquoi nous choisir ?',
      'home.feature_accuracy_title': 'Haute précision',
      'home.feature_accuracy_desc': 'Moteur hybride IA + règles avec seuils réglables pour une détection fiable.',
      'home.feature_fast_title': 'Réponse rapide',
      'home.feature_fast_desc': 'Inférence à faible latence avec cache et limites par plan.',
      'home.feature_ai_title': 'Propulsé par l’IA',
      'home.feature_ai_desc': 'Heuristiques avancées pour liens suspects, mots interdits et ponctuation excessive.',
      'home.examples_title': 'Messages réels. Détection réelle.',
      'home.examples.input': 'Entrée',
      'home.examples.result': 'Résultat',

      'pricing.title': 'Tarifs simples et transparents',
      'pricing.subtitle': 'Passez du hobby à la production. Surclassement à tout moment sur RapidAPI.',
      'pricing.note': 'Les prix et limites sont gérés sur RapidAPI. Taxes éventuelles.',
      'pricing.req_per_month': 'req / mois',
      'pricing.req_per_minute': 'req / minute',
      'plan.free': 'Gratuit',
      'plan.start_free': 'Commencer',
      'plan.pro5k': 'Pro 5k',
      'plan.pro15k': 'Pro 15k',
      'plan.pro50k': 'Pro 50k',
      'plan.choose_pro5k': 'Choisir Pro 5k',
      'plan.choose_pro15k': 'Choisir Pro 15k',
      'plan.choose_pro50k': 'Choisir Pro 50k',

      'docs.title': 'Documentation API',
      'docs.endpoints': 'Endpoints',
      'docs.examples': 'Exemples',
      'docs.errors': 'Codes d’erreur',
      'docs.pricing': 'Tarifs',
      'errors.400': 'Payload invalide. Le body doit contenir une chaîne "text" non vide.',
      'errors.401': 'Clé API manquante ou invalide.',
      'errors.403': 'Clé non abonnée à un plan ou accès refusé.',
      'errors.429': 'Limite de débit dépassée. Respectez les limites par minute selon le plan.',
      'errors.500': 'Erreur inattendue. Réessayez ou contactez le support.',

      'play.title': 'Playground API',
      'play.key': 'Clé RapidAPI',
      'play.host': 'Hôte RapidAPI',
      'play.base_url': 'URL de base',
      'play.use_proxy': 'Utiliser la clé de démo cachée (proxy sécurisé)',
      'play.use_proxy_yes': 'Oui (recommandé)',
      'play.use_proxy_no': 'Non (je fournis ma clé)',
      'play.text_label': 'Texte à analyser',
      'play.btn_check': 'Vérifier le spam',
      'play.btn_save': 'Enregistrer',
      'play.btn_clear': 'Effacer',
      'play.json_response': 'Réponse JSON',
      'play.msg.enter_text': 'Veuillez saisir un texte à analyser.',
      'play.msg.sending': 'Envoi de la requête…',
      'play.msg.saved': 'Paramètres enregistrés localement.',
      'play.msg.success': 'Succès',
      'play.msg.error': 'Requête terminée avec des erreurs',
      'play.msg.network': 'Erreur réseau',

      'contact.title': 'Contact & Support',
      'contact.name': 'Nom',
      'contact.email': 'Email',
      'contact.message': 'Message',
      'contact.send': 'Envoyer',
      'contact.github': 'Dépôt',
    }
  };

  function getPreferred() {
    const saved = localStorage.getItem('siteLang');
    if (saved && (saved === 'en' || saved === 'fr')) return saved;
    const nav = (navigator.language || 'en').toLowerCase();
    return nav.startsWith('fr') ? 'fr' : 'en';
  }

  function t(key, lang) {
    const l = lang || I18N.current;
    return (dict[l] && dict[l][key]) || (dict.en && dict.en[key]) || key;
  }

  function apply(lang) {
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      el.innerHTML = t(key, lang);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      el.setAttribute('placeholder', t(key, lang));
    });
    document.querySelectorAll('[data-i18n-value]').forEach(el => {
      const key = el.getAttribute('data-i18n-value');
      el.setAttribute('value', t(key, lang));
    });
    const sel = document.getElementById('langSelect');
    if (sel) sel.value = lang;
    window.dispatchEvent(new CustomEvent('i18n:changed', { detail: { lang } }));
  }

  const I18N = {
    current: getPreferred(),
    t,
    set(lang) { this.current = lang; localStorage.setItem('siteLang', lang); apply(lang); },
    init() { apply(this.current); }
  };

  window.I18N = I18N;
})();


