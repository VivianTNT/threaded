/**
 * Advanced Mining Metrics Extractor
 * Validates documents have 40% of key terms and extracts financial values
 */

export interface MiningMetric {
  category: string;
  field: string;
  canonicalTerm: string;
  keywords: string[];
  regex?: RegExp;
  units?: string;
  value?: any;
  found?: boolean;
}

// Complete list of mining metrics with extraction patterns
export const MINING_METRICS: MiningMetric[] = [
  // Economic Metrics
  {
    category: 'Economic Metrics',
    field: 'NPV',
    canonicalTerm: 'NPV (post-tax)',
    keywords: ['NPV after tax', 'post-tax NPV', 'NPV (AT)', 'after-tax net present value'],
    regex: /(?:post[\s-]?tax|after[\s-]?tax)\s*NPV[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:M|million|MM|B|billion)/gi,
    units: 'USD millions'
  },
  {
    category: 'Economic Metrics',
    field: 'NPV',
    canonicalTerm: 'NPV (pre-tax)',
    keywords: ['NPV pre tax', 'pre-tax NPV', 'NPV (BT)', 'before-tax NPV'],
    regex: /(?:pre[\s-]?tax|before[\s-]?tax)\s*NPV[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:M|million|MM|B|billion)/gi,
    units: 'USD millions'
  },
  {
    category: 'Economic Metrics',
    field: 'IRR',
    canonicalTerm: 'IRR (post-tax)',
    keywords: ['IRR after tax', 'post-tax IRR', 'IRR (AT)', 'after-tax internal rate'],
    regex: /(?:post[\s-]?tax|after[\s-]?tax)\s*IRR[^\d]*([\d.]+)\s*%/gi,
    units: 'percent'
  },
  {
    category: 'Economic Metrics',
    field: 'IRR',
    canonicalTerm: 'IRR (pre-tax)',
    keywords: ['IRR pre tax', 'pre-tax IRR', 'IRR (BT)', 'before-tax IRR'],
    regex: /(?:pre[\s-]?tax|before[\s-]?tax)\s*IRR[^\d]*([\d.]+)\s*%/gi,
    units: 'percent'
  },
  {
    category: 'Economic Metrics',
    field: 'Payback Period',
    canonicalTerm: 'Payback Period',
    keywords: ['payback', 'payback period', 'capital payback', 'investment payback'],
    regex: /payback(?:\s+period)?[^\d]*([\d.]+)\s*(?:years?|yrs?)/gi,
    units: 'years'
  },

  // Capital Costs
  {
    category: 'Capital Costs',
    field: 'Initial CAPEX',
    canonicalTerm: 'Initial CAPEX',
    keywords: ['initial capital', 'CAPEX', 'capital expenditure', 'pre-production capital', 'development capital'],
    regex: /(?:initial|pre[\s-]?production|development)\s*(?:capital|CAPEX)[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:M|million|MM|B|billion)/gi,
    units: 'USD millions'
  },
  {
    category: 'Capital Costs',
    field: 'Sustaining CAPEX',
    canonicalTerm: 'Sustaining CAPEX',
    keywords: ['sustaining capital', 'sustaining CAPEX', 'LOM sustaining', 'maintenance capital'],
    regex: /sustaining\s*(?:capital|CAPEX)[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:M|million|MM|B|billion)/gi,
    units: 'USD millions'
  },

  // Operating Costs
  {
    category: 'Operating Costs',
    field: 'OPEX',
    canonicalTerm: 'Operating Cost (OPEX)',
    keywords: ['operating cost', 'OPEX', 'operating costs', 'site operating cost'],
    regex: /(?:operating\s*cost|OPEX)[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:\/|per)\s*(?:tonne?|t|ton)/gi,
    units: 'USD/tonne'
  },
  {
    category: 'Operating Costs',
    field: 'AISC',
    canonicalTerm: 'AISC',
    keywords: ['AISC', 'all-in sustaining cost', 'all in sustaining'],
    regex: /AISC[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:\/|per)\s*(?:oz|ounce|lb|pound|tonne?|t)/gi,
    units: 'USD/unit'
  },
  {
    category: 'Operating Costs',
    field: 'Cash Cost',
    canonicalTerm: 'Cash Cost',
    keywords: ['cash cost', 'cash costs', 'site cash cost'],
    regex: /cash\s*cost[^\d]*([\$US]*[\s]*[\d,]+(?:\.\d+)?)[\s]*(?:\/|per)\s*(?:oz|ounce|lb|pound|tonne?|t)/gi,
    units: 'USD/unit'
  },

  // Mine Profile
  {
    category: 'Mine Profile',
    field: 'Mine Life',
    canonicalTerm: 'Mine Life',
    keywords: ['mine life', 'life of mine', 'LOM', 'project life'],
    regex: /(?:mine\s*life|life\s*of\s*mine|LOM)[^\d]*([\d.]+)\s*(?:years?|yrs?)/gi,
    units: 'years'
  },
  {
    category: 'Mine Profile',
    field: 'Annual Production',
    canonicalTerm: 'Annual Production',
    keywords: ['annual production', 'yearly production', 'average annual production'],
    regex: /(?:annual|yearly|average\s*annual)\s*production[^\d]*([\d,]+(?:\.\d+)?)[\s]*(?:k?t|tonnes?|tons?|Mtpa|oz|ounces|lbs?|pounds?)/gi,
    units: 'varies'
  },
  {
    category: 'Mine Profile',
    field: 'Throughput',
    canonicalTerm: 'Throughput',
    keywords: ['throughput', 'nameplate capacity', 'processing rate', 'tpd', 'Mtpa'],
    regex: /(?:throughput|processing\s*rate|nameplate)[^\d]*([\d,]+(?:\.\d+)?)[\s]*(?:Mt?pa|tpd|t\/d|tonnes?\/day)/gi,
    units: 'tpd or Mtpa'
  },

  // Resources & Reserves
  {
    category: 'Resources',
    field: 'Total Resources',
    canonicalTerm: 'Total Resources',
    keywords: ['mineral resources', 'total resources', 'measured and indicated', 'M&I', 'inferred'],
    regex: /(?:total|mineral)\s*resources?[^\d]*([\d,]+(?:\.\d+)?)[\s]*(?:Mt|million\s*tonnes?|million\s*tons?)/gi,
    units: 'million tonnes'
  },
  {
    category: 'Resources',
    field: 'Grade',
    canonicalTerm: 'Grade',
    keywords: ['grade', 'average grade', 'resource grade', 'ore grade'],
    regex: /(?:average|ore|resource)?\s*grade[^\d]*([\d.]+)[\s]*(?:g\/t|%|ppm|oz\/t)/gi,
    units: 'varies'
  },
  {
    category: 'Resources',
    field: 'Reserves',
    canonicalTerm: 'Reserves',
    keywords: ['ore reserves', 'proven and probable', 'P&P', 'mineral reserves'],
    regex: /(?:ore|mineral)\s*reserves?[^\d]*([\d,]+(?:\.\d+)?)[\s]*(?:Mt|million\s*tonnes?|million\s*tons?)/gi,
    units: 'million tonnes'
  },

  // Recovery
  {
    category: 'Metallurgy',
    field: 'Recovery',
    canonicalTerm: 'Metallurgical Recovery',
    keywords: ['recovery', 'metallurgical recovery', 'processing recovery', 'overall recovery'],
    regex: /(?:metallurgical|processing|overall)?\s*recovery[^\d]*([\d.]+)[\s]*%/gi,
    units: 'percent'
  },

  // Project Stage
  {
    category: 'Project Overview',
    field: 'Project Stage',
    canonicalTerm: 'Project Stage',
    keywords: ['exploration', 'PEA', 'preliminary economic assessment', 'pre-feasibility', 'PFS', 'feasibility study', 'DFS', 'BFS', 'construction', 'production'],
    regex: /\b(exploration|PEA|preliminary\s*economic\s*assessment|pre[\s-]?feasibility|PFS|feasibility\s*study|DFS|BFS|construction|production)\b/gi,
    units: 'stage'
  },

  // Compliance
  {
    category: 'Compliance',
    field: 'Reporting Code',
    canonicalTerm: 'Reporting Code',
    keywords: ['NI 43-101', 'S-K 1300', 'SK1300', 'JORC', 'SAMREC'],
    regex: /\b(NI\s*43[\s-]101|S[\s-]K\s*1300|SK1300|JORC|SAMREC)\b/gi,
    units: 'standard'
  }
];

/**
 * Extract numeric value from text with unit conversion
 */
function extractNumericValue(text: string, pattern: RegExp): number | null {
  const match = pattern.exec(text);
  if (!match || !match[1]) return null;

  // Clean the value
  let valueStr = match[1].replace(/[\$,US\s]/g, '');
  let value = parseFloat(valueStr);

  if (isNaN(value)) return null;

  // Convert billions to millions if needed
  if (match[0].toLowerCase().includes('billion') || match[0].includes('B')) {
    value = value * 1000;
  }

  return value;
}

/**
 * Validate if document contains minimum required metrics (40% threshold)
 */
export function validateDocumentMetrics(text: string): {
  isValid: boolean;
  metricsFound: number;
  totalMetrics: number;
  percentage: number;
  foundMetrics: string[];
} {
  const foundMetrics: string[] = [];
  let metricsFound = 0;

  for (const metric of MINING_METRICS) {
    let found = false;

    // Check keywords
    for (const keyword of metric.keywords) {
      if (text.toLowerCase().includes(keyword.toLowerCase())) {
        found = true;
        break;
      }
    }

    // Check regex if available
    if (!found && metric.regex) {
      const match = metric.regex.test(text);
      if (match) found = true;
    }

    if (found) {
      metricsFound++;
      foundMetrics.push(metric.canonicalTerm);
    }
  }

  const totalMetrics = MINING_METRICS.length;
  const percentage = (metricsFound / totalMetrics) * 100;

  return {
    isValid: percentage >= 40,
    metricsFound,
    totalMetrics,
    percentage: Math.round(percentage),
    foundMetrics
  };
}

/**
 * Extract all financial metrics from document text
 */
export function extractFinancialMetrics(text: string): Record<string, any> {
  const extracted: Record<string, any> = {};

  // NPV Post-tax
  const npvPostTaxPattern = /(?:post[\s-]?tax|after[\s-]?tax)\s*NPV[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:M|million|MM)/gi;
  const npvPostTax = extractNumericValue(text, npvPostTaxPattern);
  if (npvPostTax) extracted.post_tax_npv_usd_m = npvPostTax;

  // NPV Pre-tax
  const npvPreTaxPattern = /(?:pre[\s-]?tax|before[\s-]?tax)\s*NPV[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:M|million|MM)/gi;
  const npvPreTax = extractNumericValue(text, npvPreTaxPattern);
  if (npvPreTax) extracted.pre_tax_npv_usd_m = npvPreTax;

  // IRR
  const irrPattern = /(?:post[\s-]?tax|after[\s-]?tax)?\s*IRR[^\d]*([\d.]+)\s*%/gi;
  const irr = extractNumericValue(text, irrPattern);
  if (irr) extracted.irr_percent = irr;

  // Payback Period
  const paybackPattern = /payback(?:\s+period)?[^\d]*([\d.]+)\s*(?:years?|yrs?)/gi;
  const payback = extractNumericValue(text, paybackPattern);
  if (payback) extracted.payback_years = payback;

  // Initial CAPEX
  const capexPattern = /(?:initial|pre[\s-]?production|development|total)\s*(?:capital|CAPEX)[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:M|million|MM)/gi;
  const capex = extractNumericValue(text, capexPattern);
  if (capex) extracted.capex_usd_m = capex;

  // Sustaining CAPEX
  const sustainingPattern = /sustaining\s*(?:capital|CAPEX)[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:M|million|MM)/gi;
  const sustaining = extractNumericValue(text, sustainingPattern);
  if (sustaining) extracted.sustaining_capex_usd_m = sustaining;

  // Mine Life
  const mineLifePattern = /(?:mine\s*life|life\s*of\s*mine|LOM)[^\d]*([\d.]+)\s*(?:years?|yrs?)/gi;
  const mineLife = extractNumericValue(text, mineLifePattern);
  if (mineLife) extracted.mine_life_years = mineLife;

  // Annual Production
  const productionPattern = /(?:annual|yearly|average\s*annual)\s*production[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|million\s*tonnes?|k?t\/year)/gi;
  const production = extractNumericValue(text, productionPattern);
  if (production) {
    // Convert to tonnes if in millions
    if (text.includes('Mt') || text.includes('million')) {
      extracted.annual_production_tonnes = production * 1000000;
    } else {
      extracted.annual_production_tonnes = production;
    }
  }

  // Total Resources
  const resourcePattern = /(?:total|mineral)\s*resources?[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|million\s*tonnes?)/gi;
  const resources = extractNumericValue(text, resourcePattern);
  if (resources) extracted.total_resource_tonnes = resources * 1000000;

  // Grade
  const gradePattern = /(?:average|ore|resource)?\s*grade[^\d]*([\d.]+)\s*(g\/t|%|ppm|oz\/t)/gi;
  const gradeMatch = gradePattern.exec(text);
  if (gradeMatch) {
    extracted.resource_grade = parseFloat(gradeMatch[1]);
    extracted.resource_grade_unit = gradeMatch[2];
  }

  // OPEX
  const opexPattern = /(?:operating\s*cost|OPEX)[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:\/|per)\s*(?:tonne?|t)/gi;
  const opex = extractNumericValue(text, opexPattern);
  if (opex) extracted.opex_usd_per_tonne = opex;

  // AISC
  const aiscPattern = /AISC[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:\/|per)\s*(?:oz|ounce|lb|pound|tonne?)/gi;
  const aisc = extractNumericValue(text, aiscPattern);
  if (aisc) extracted.aisc_usd_per_tonne = aisc;

  // Project Stage
  const stagePattern = /\b(exploration|PEA|preliminary\s*economic\s*assessment|pre[\s-]?feasibility|feasibility\s*study|construction|production)\b/gi;
  const stageMatch = stagePattern.exec(text);
  if (stageMatch) {
    const stage = stageMatch[1].toLowerCase();
    if (stage.includes('exploration')) extracted.stage = 'exploration';
    else if (stage.includes('pea') || stage.includes('preliminary')) extracted.stage = 'pea';
    else if (stage.includes('pre-feasibility') || stage.includes('prefeasibility')) extracted.stage = 'pre_feasibility';
    else if (stage.includes('feasibility') && !stage.includes('pre')) extracted.stage = 'feasibility';
    else if (stage.includes('construction')) extracted.stage = 'construction';
    else if (stage.includes('production')) extracted.stage = 'production';
  }

  return extracted;
}

/**
 * Calculate extraction confidence based on metrics found
 */
export function calculateExtractionConfidence(extractedMetrics: Record<string, any>): number {
  const criticalMetrics = [
    'post_tax_npv_usd_m',
    'capex_usd_m',
    'irr_percent',
    'mine_life_years',
    'annual_production_tonnes'
  ];

  let foundCount = 0;
  for (const metric of criticalMetrics) {
    if (extractedMetrics[metric] !== undefined) {
      foundCount++;
    }
  }

  // Base confidence on critical metrics found
  let confidence = (foundCount / criticalMetrics.length) * 60;

  // Add bonus for additional metrics
  const bonusMetrics = [
    'pre_tax_npv_usd_m',
    'sustaining_capex_usd_m',
    'payback_years',
    'total_resource_tonnes',
    'resource_grade',
    'opex_usd_per_tonne',
    'aisc_usd_per_tonne'
  ];

  let bonusCount = 0;
  for (const metric of bonusMetrics) {
    if (extractedMetrics[metric] !== undefined) {
      bonusCount++;
    }
  }

  confidence += (bonusCount / bonusMetrics.length) * 40;

  return Math.min(Math.round(confidence), 100);
}