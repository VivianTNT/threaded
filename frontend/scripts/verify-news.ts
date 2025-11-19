import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '../.env.local') });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function verifyNews() {
  const { data, error, count } = await supabase
    .from('news')
    .select('*', { count: 'exact' });

  if (error) {
    console.error('Error fetching news:', error);
    return;
  }

  console.log(`Total news articles in database: ${count}`);
  console.log('\nSample articles:');

  if (data && data.length > 0) {
    data.slice(0, 5).forEach((article: any, idx: number) => {
      console.log(`\n${idx + 1}. ${article.title}`);
      console.log(`   Source: ${article.source}`);
      console.log(`   Published: ${article.published_at}`);
      console.log(`   Commodities: ${article.commodities?.join(', ')}`);
      console.log(`   Sentiment: ${article.sentiment}`);
    });
  }
}

verifyNews().catch(console.error);
