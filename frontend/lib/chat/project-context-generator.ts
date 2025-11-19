// Project Context Generator for Chat Integration
// Generates comprehensive project summaries with risk analysis and predictive analytics

import { MiningProject } from '@/lib/types/mining-project'

export interface ProjectSummary {
  overview: string
  keyMetrics: string
  riskFactors: RiskAnalysis
  opportunities: string[]
  comparisons: string
  recommendations: string
}

export interface RiskAnalysis {
  geological: { level: string; factors: string[] }
  market: { level: string; factors: string[] }
  operational: { level: string; factors: string[] }
  regulatory: { level: string; factors: string[] }
  financial: { level: string; factors: string[] }
  overall: string
}

export interface NewsAnalysis {
  summary: string
  sentiment: 'positive' | 'neutral' | 'negative'
  riskFactors: string[]
  opportunities: string[]
  predictiveInsights: {
    priceImpact: string
    investmentRecommendation: string
    marketTrends: string[]
    competitiveAdvantage: string
  }
}

export function generateProjectSummary(project: MiningProject): ProjectSummary {
  const overview = `
**${project.project || project.name}** 
Company: ${project.company || 'N/A'}
Location: ${project.jurisdiction}, ${project.country || 'N/A'}
Stage: ${project.stage}
Primary Commodity: ${project.primaryCommodity}

This ${project.stage}-stage ${project.primaryCommodity} project ${
    project.mineLife ? `has an estimated mine life of ${project.mineLife} years` : 'is in early development'
  }.`;

  const keyMetrics = `
**Key Financial Metrics:**
• NPV (Post-tax): $${project.postTaxNPV || 0}M
• IRR: ${project.irr || 0}%
• CAPEX: $${project.capex || 0}M
• AISC: $${project.aisc || 0}/t
• Payback Period: ${project.paybackYears || 'N/A'} years
• Mine Life: ${project.mineLife || 'N/A'} years`;

  const riskFactors = generateRiskAnalysis(project);
  const opportunities = generateOpportunities(project);
  const comparisons = generateComparisons(project);
  const recommendations = generateRecommendations(project, riskFactors);

  return {
    overview,
    keyMetrics,
    riskFactors,
    opportunities,
    comparisons,
    recommendations
  };
}

function generateRiskAnalysis(project: MiningProject): RiskAnalysis {
  const geological: RiskAnalysis['geological'] = {
    level: project.stage === 'Exploration' ? 'High' : project.stage === 'Production' ? 'Low' : 'Medium',
    factors: []
  };

  const market: RiskAnalysis['market'] = {
    level: 'Medium',
    factors: []
  };

  const operational: RiskAnalysis['operational'] = {
    level: 'Medium',
    factors: []
  };

  const regulatory: RiskAnalysis['regulatory'] = {
    level: project.riskLevel || 'Medium',
    factors: []
  };

  const financial: RiskAnalysis['financial'] = {
    level: project.irr && project.irr > 20 ? 'Low' : project.irr && project.irr > 15 ? 'Medium' : 'High',
    factors: []
  };

  // Geological risks
  if (project.stage === 'Exploration' || project.stage === 'Resource Definition') {
    geological.factors.push('Resource uncertainty - further drilling required');
    geological.factors.push('Grade variability risk');
  }
  if (!project.resourceGrade) {
    geological.factors.push('Grade information not available');
  }

  // Market risks
  market.factors.push(`${project.primaryCommodity} price volatility`);
  if (project.primaryCommodity === 'Lithium' || project.primaryCommodity === 'Cobalt') {
    market.factors.push('EV market dependency');
    market.factors.push('Technology substitution risk');
  }

  // Operational risks
  if (project.capex && project.capex > 1000) {
    operational.factors.push('High capital intensity');
  }
  if (project.aisc && project.aisc > 100) {
    operational.factors.push('Above-average operating costs');
  }
  if (project.jurisdiction && ['DRC', 'Russia', 'Venezuela'].includes(project.jurisdiction)) {
    operational.factors.push('Challenging operating environment');
  }

  // Regulatory risks
  if (project.permitStatus !== 'Granted') {
    regulatory.factors.push('Permitting not complete');
  }
  if (project.esgScore === 'D' || project.esgScore === 'F') {
    regulatory.factors.push('ESG compliance concerns');
  }
  regulatory.factors.push(`Jurisdiction risk: ${project.riskLevel || 'Medium'}`);

  // Financial risks
  if (project.paybackYears && project.paybackYears > 5) {
    financial.factors.push('Extended payback period');
  }
  if (project.irr && project.irr < 15) {
    financial.factors.push('Below industry average returns');
  }
  if (project.capex && project.capex > 500) {
    financial.factors.push('Significant capital requirements');
  }

  // Calculate overall risk
  const riskLevels = [geological.level, market.level, operational.level, regulatory.level, financial.level];
  const highCount = riskLevels.filter(l => l === 'High').length;
  const overall = highCount >= 3 ? 'High' : highCount >= 1 ? 'Medium' : 'Low';

  return {
    geological,
    market,
    operational,
    regulatory,
    financial,
    overall
  };
}

function generateOpportunities(project: MiningProject): string[] {
  const opportunities: string[] = [];

  if (project.irr && project.irr > 25) {
    opportunities.push('High IRR indicates strong investment potential');
  }

  if (project.stage === 'Exploration' || project.stage === 'Resource Definition') {
    opportunities.push('Early-stage entry opportunity with potential for value appreciation');
  }

  if (project.primaryCommodity === 'Lithium' || project.primaryCommodity === 'Copper' || project.primaryCommodity === 'Nickel') {
    opportunities.push('Critical mineral for energy transition');
  }

  if (project.mineLife && project.mineLife > 20) {
    opportunities.push('Long mine life provides stable long-term cash flows');
  }

  if (project.paybackYears && project.paybackYears < 3) {
    opportunities.push('Quick payback period reduces investment risk');
  }

  return opportunities;
}

function generateComparisons(project: MiningProject): string {
  const avgIRR = 20; // Industry average
  const avgPayback = 4; // Industry average years

  let comparison = `Compared to industry averages:\n`;
  
  if (project.irr) {
    const irrDiff = project.irr - avgIRR;
    comparison += `• IRR is ${Math.abs(irrDiff).toFixed(1)}% ${irrDiff > 0 ? 'above' : 'below'} average (${avgIRR}%)\n`;
  }

  if (project.paybackYears) {
    const paybackDiff = project.paybackYears - avgPayback;
    comparison += `• Payback period is ${Math.abs(paybackDiff).toFixed(1)} years ${paybackDiff > 0 ? 'longer' : 'shorter'} than average\n`;
  }

  if (project.stage === 'Production') {
    comparison += `• Project is in production, reducing development risk\n`;
  }

  return comparison;
}

function generateRecommendations(project: MiningProject, risks: RiskAnalysis): string {
  const recommendations: string[] = [];

  if (risks.overall === 'Low' && project.irr && project.irr > 20) {
    recommendations.push('Strong candidate for investment consideration');
  } else if (risks.overall === 'High') {
    recommendations.push('High-risk project requiring careful due diligence');
  } else {
    recommendations.push('Moderate risk-return profile suitable for diversified portfolios');
  }

  if (project.stage === 'Exploration') {
    recommendations.push('Monitor resource definition progress and drilling results');
  }

  if (risks.regulatory.factors.includes('Permitting not complete')) {
    recommendations.push('Track permitting milestones closely');
  }

  if (project.primaryCommodity === 'Lithium' || project.primaryCommodity === 'Copper') {
    recommendations.push('Consider as part of energy transition investment strategy');
  }

  return recommendations.join('. ') + '.';
}

export function generateNewsAnalysis(news: any): NewsAnalysis {
  // Analyze sentiment based on headline and content
  const positiveKeywords = ['record', 'increase', 'growth', 'expansion', 'success', 'breakthrough', 'approval', 'discovery'];
  const negativeKeywords = ['decline', 'delay', 'risk', 'concern', 'challenge', 'failure', 'rejection', 'closure'];
  
  const headlineText = (news.headline || '').toLowerCase();
  const summaryText = (news.summary || '').toLowerCase();
  const combinedText = headlineText + ' ' + summaryText;

  const positiveCount = positiveKeywords.filter(word => combinedText.includes(word)).length;
  const negativeCount = negativeKeywords.filter(word => combinedText.includes(word)).length;

  const sentiment: NewsAnalysis['sentiment'] = 
    positiveCount > negativeCount ? 'positive' : 
    negativeCount > positiveCount ? 'negative' : 
    'neutral';

  // Generate risk factors
  const riskFactors: string[] = [];
  if (combinedText.includes('delay')) riskFactors.push('Project timeline delays');
  if (combinedText.includes('cost')) riskFactors.push('Cost overrun potential');
  if (combinedText.includes('permit')) riskFactors.push('Regulatory approval risks');
  if (combinedText.includes('price')) riskFactors.push('Commodity price volatility');

  // Generate opportunities
  const opportunities: string[] = [];
  if (combinedText.includes('discovery')) opportunities.push('Resource expansion potential');
  if (combinedText.includes('approval')) opportunities.push('Development milestone achieved');
  if (combinedText.includes('production')) opportunities.push('Revenue generation approaching');
  if (combinedText.includes('partnership')) opportunities.push('Strategic partnership benefits');

  // Predictive insights
  const predictiveInsights = {
    priceImpact: generatePriceImpact(news, sentiment),
    investmentRecommendation: generateInvestmentRec(news, sentiment),
    marketTrends: generateMarketTrends(news),
    competitiveAdvantage: generateCompetitiveAdvantage(news)
  };

  return {
    summary: `${news.headline}. ${news.summary || ''}`.substring(0, 200),
    sentiment,
    riskFactors,
    opportunities,
    predictiveInsights
  };
}

function generatePriceImpact(news: any, sentiment: string): string {
  const commodity = news.primary_commodity || 'commodity';
  
  if (sentiment === 'positive') {
    if (news.is_project_related) {
      return `Potential positive impact on ${commodity} demand and pricing in the medium term`;
    }
    return `Bullish signal for ${commodity} prices`;
  } else if (sentiment === 'negative') {
    return `May create short-term pressure on ${commodity} prices`;
  }
  
  return `Neutral impact on ${commodity} market pricing expected`;
}

function generateInvestmentRec(news: any, sentiment: string): string {
  if (sentiment === 'positive' && news.mentions_financials) {
    return 'Consider increasing exposure to related mining assets';
  } else if (sentiment === 'negative') {
    return 'Monitor situation closely before making investment decisions';
  }
  
  return 'Maintain current investment position while monitoring developments';
}

function generateMarketTrends(news: any): string[] {
  const trends: string[] = [];
  
  if (news.primary_commodity === 'lithium') {
    trends.push('EV market growth driving demand');
    trends.push('Battery technology evolution');
  } else if (news.primary_commodity === 'copper') {
    trends.push('Infrastructure investment cycle');
    trends.push('Green energy transition demand');
  } else if (news.primary_commodity === 'gold') {
    trends.push('Safe haven demand dynamics');
    trends.push('Central bank purchasing trends');
  }
  
  if (news.is_project_related) {
    trends.push('Project pipeline development');
  }
  
  return trends;
}

function generateCompetitiveAdvantage(news: any): string {
  if (news.mentions_technical_report) {
    return 'Technical validation provides credibility advantage';
  } else if (news.is_project_related && news.sentiment_score > 0.7) {
    return 'Positive project developments strengthen market position';
  } else if (news.company_name) {
    return `${news.company_name} positioning for market opportunities`;
  }
  
  return 'Market positioning remains stable';
}

export function formatProjectForChat(project: MiningProject): string {
  const summary = generateProjectSummary(project);
  
  return `
${summary.overview}

${summary.keyMetrics}

**Risk Analysis:**
• Geological Risk: ${summary.riskFactors.geological.level}
  ${summary.riskFactors.geological.factors.map(f => `  - ${f}`).join('\n')}
• Market Risk: ${summary.riskFactors.market.level}
  ${summary.riskFactors.market.factors.map(f => `  - ${f}`).join('\n')}
• Operational Risk: ${summary.riskFactors.operational.level}
  ${summary.riskFactors.operational.factors.map(f => `  - ${f}`).join('\n')}
• Regulatory Risk: ${summary.riskFactors.regulatory.level}
  ${summary.riskFactors.regulatory.factors.map(f => `  - ${f}`).join('\n')}
• Financial Risk: ${summary.riskFactors.financial.level}
  ${summary.riskFactors.financial.factors.map(f => `  - ${f}`).join('\n')}

**Overall Risk Assessment:** ${summary.riskFactors.overall}

**Investment Opportunities:**
${summary.opportunities.map(o => `• ${o}`).join('\n')}

**Peer Comparison:**
${summary.comparisons}

**Recommendations:**
${summary.recommendations}
`;
}

export function formatNewsForChat(news: any): string {
  const analysis = generateNewsAnalysis(news);
  
  return `
**News Article:** ${news.headline}
**Date:** ${news.datetime || news.published_date}
**Source:** ${news.source}
**Commodity:** ${news.primary_commodity}
**Sentiment:** ${analysis.sentiment}

**Summary:**
${analysis.summary}

**Risk Factors:**
${analysis.riskFactors.map(r => `• ${r}`).join('\n')}

**Opportunities:**
${analysis.opportunities.map(o => `• ${o}`).join('\n')}

**Predictive Analytics:**
• **Price Impact:** ${analysis.predictiveInsights.priceImpact}
• **Investment Recommendation:** ${analysis.predictiveInsights.investmentRecommendation}
• **Competitive Advantage:** ${analysis.predictiveInsights.competitiveAdvantage}

**Market Trends:**
${analysis.predictiveInsights.marketTrends.map(t => `• ${t}`).join('\n')}

**Investment Perspective:**
This news ${analysis.sentiment === 'positive' ? 'presents opportunities for' : analysis.sentiment === 'negative' ? 'suggests caution in' : 'maintains neutral outlook for'} mining investments. ${analysis.predictiveInsights.investmentRecommendation}
`;
}
