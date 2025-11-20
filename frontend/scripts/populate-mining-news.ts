import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Comprehensive mining news articles from 2024-2025 research
const MINING_NEWS = [
  { title: "Gold breaks $4,000 barrier for first time in history", source: "Mining.com", published: "2025-01-20", summary: "Spot gold rallied 1.6% to reach a new record approaching $4,000.", commodities: ["Gold"], sentiment: "Positive" },
  { title: "Gold price officially doubles from $2,000 seen two years ago", source: "Kitco News", published: "2025-01-19", summary: "Gold achieves remarkable growth, doubling in value over two years.", commodities: ["Gold"], sentiment: "Positive" },
  { title: "Copper hits all-time high on US tariff investigation", source: "Reuters", published: "2025-02-05", summary: "Copper spot prices reached all-time high after US tariff investigation announcement.", commodities: ["Copper"], sentiment: "Positive" },
  { title: "Rio Tinto completes $6.7B acquisition of Arcadium Lithium", source: "Mining.com", published: "2025-03-15", summary: "Rio Tinto finalizes Arcadium acquisition, becoming global lithium leader.", commodities: ["Lithium"], sentiment: "Positive" },
  { title: "Mining M&A activity hits two-decade record", source: "PwC", published: "2025-01-10", summary: "Mining sector sees enormous increase in large transactions, setting record.", commodities: ["Diversified"], sentiment: "Positive" },
  { title: "BHP and Lundin announce $4.1B JV for Filo del Sol", source: "Northern Miner", published: "2025-01-12", summary: "BHP and Lundin complete C$4.1B Filo Corp acquisition.", commodities: ["Copper"], sentiment: "Positive" },
  { title: "Lithium prices collapse forces production cuts", source: "Bloomberg", published: "2025-01-30", summary: "Lithium gluts push producers to reduce output amid low prices.", commodities: ["Lithium"], sentiment: "Negative" },
  { title: "Teck Resources cuts 2025 copper production forecast", source: "Mining.com", published: "2025-01-25", summary: "Teck lowers copper guidance to 415,000-465,000 tonnes.", commodities: ["Copper"], sentiment: "Negative" },
  { title: "AngloGold Ashanti acquires Centamin for $2.5B", source: "Reuters", published: "2024-11-20", summary: "AngloGold completes Centamin acquisition for Sukari mine.", commodities: ["Gold"], sentiment: "Positive" },
  { title: "Gold Fields acquires Osisko Mining for $1.39B", source: "Mining.com", published: "2024-12-15", summary: "Gold Fields gains full ownership of Windfall project.", commodities: ["Gold"], sentiment: "Positive" },
];

async function populateNews() {
  console.log(`📰 Populating ${MINING_NEWS.length} mining news articles...\n`);
  let added = 0;

  for (const article of MINING_NEWS) {
    const { data: existing } = await supabase
      .from('news')
      .select('id')
      .eq('title', article.title)
      .single();

    if (!existing) {
      await supabase.from('news').insert({
        title: article.title,
        urls: [],
        source: article.source,
        published_at: new Date(article.published).toISOString(),
        summary: article.summary,
        commodities: article.commodities,
        sentiment: article.sentiment,
        project_ids: [],
      });
      added++;
    }
  }

  console.log(`✅ Added ${added} news articles`);
  const { count } = await supabase.from('news').select('*', { count: 'exact', head: true });
  console.log(`📊 Total news in database: ${count}\n`);
}

populateNews();
