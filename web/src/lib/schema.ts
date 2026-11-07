// JSON-LD structured data for the public pages: Organization, WebSite,
// SoftwareApplication, FAQPage and Article. Search engines and AI answers
// engines use it to understand and cite the site. Keep every value consistent
// with the visible page content — never claim anything the page doesn't say.
import { siteOrigin } from "./seo";

const NAME = "HSF AI";
const SITE = "HSFinest.com";
const GITHUB = "https://github.com/realhowieb/ai_scanner";

/** Who runs the site. One Organization block, referenced by the other schemas. */
export function organizationSchema() {
  const origin = siteOrigin();
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    "@id": `${origin}/#organization`,
    name: SITE,
    url: origin,
    logo: `${origin}/icon.svg`,
    sameAs: [GITHUB],
  };
}

/** The site itself. */
export function webSiteSchema() {
  const origin = siteOrigin();
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "@id": `${origin}/#website`,
    name: SITE,
    url: origin,
    publisher: { "@id": `${origin}/#organization` },
  };
}

/**
 * The product. `offers` only asserts the free Starter plan — paid plan prices
 * come from GET /v1/plans at runtime and change, so they are not claimed here.
 */
export function softwareApplicationSchema(description: string) {
  const origin = siteOrigin();
  return {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "@id": `${origin}/#app`,
    name: NAME,
    url: origin,
    description,
    applicationCategory: "FinanceApplication",
    operatingSystem: "Web browser",
    publisher: { "@id": `${origin}/#organization` },
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
      description: "Free Starter plan: ranked Scanner, stock pages, watchlists and S&P 500 scans.",
    },
  };
}

/** A page's questions and answers, for FAQ rich results. */
export function faqSchema(pairs: readonly [string, string][]) {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: pairs.map(([q, a]) => ({
      "@type": "Question",
      name: q,
      acceptedAnswer: { "@type": "Answer", text: a },
    })),
  };
}

/** The weekly recap, as a dated article. */
export function articleSchema(opts: { headline: string; description: string; path: string; datePublished: string; dateModified?: string }) {
  const origin = siteOrigin();
  return {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: opts.headline,
    description: opts.description,
    url: `${origin}${opts.path}`,
    mainEntityOfPage: `${origin}${opts.path}`,
    datePublished: opts.datePublished,
    ...(opts.dateModified ? { dateModified: opts.dateModified } : {}),
    publisher: { "@id": `${origin}/#organization` },
    isPartOf: { "@id": `${origin}/#website` },
  };
}
