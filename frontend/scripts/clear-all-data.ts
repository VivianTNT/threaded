import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function clearAllData() {
  console.log('Clearing all existing data...');

  // Clear pdf_highlights first (foreign key constraint)
  const { error: pdfError } = await supabase
    .from('pdf_highlights')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all

  if (pdfError) {
    console.error('Error clearing pdf_highlights:', pdfError);
  } else {
    console.log('✓ Cleared pdf_highlights');
  }

  // Clear projects
  const { error: projectError } = await supabase
    .from('projects')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all

  if (projectError) {
    console.error('Error clearing projects:', projectError);
  } else {
    console.log('✓ Cleared projects');
  }

  // Clear companies
  const { error: companyError } = await supabase
    .from('companies')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all

  if (companyError) {
    console.error('Error clearing companies:', companyError);
  } else {
    console.log('✓ Cleared companies');
  }

  console.log('Data clearing complete!');
}

clearAllData().catch(console.error);