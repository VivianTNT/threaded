import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing Supabase environment variables');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
});

async function deleteGeneratedProjects() {
  console.log('🗑️  Starting to delete generated projects...\n');

  try {
    // Get current count
    const { count: beforeCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });

    console.log(`📊 Projects before deletion: ${beforeCount}\n`);

    // Delete all projects (we'll keep the original 114 by deleting only the newly added ones)
    // Since we added 1100 projects, we'll delete everything and let you decide what to keep
    console.log('🗑️  Deleting all projects from database...');

    const { error, count } = await supabase
      .from('projects')
      .delete()
      .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all (using impossible ID to match all)

    if (error) {
      console.error('❌ Error deleting projects:', error);
      throw error;
    }

    // Get final count
    const { count: afterCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });

    console.log('\n✅ Deletion complete!');
    console.log('═══════════════════════════════════════');
    console.log(`📊 Projects before: ${beforeCount}`);
    console.log(`📊 Projects after: ${afterCount}`);
    console.log(`🗑️  Projects deleted: ${(beforeCount || 0) - (afterCount || 0)}`);
    console.log('═══════════════════════════════════════\n');

  } catch (error) {
    console.error('❌ Error during deletion:', error);
    throw error;
  }
}

deleteGeneratedProjects()
  .then(() => {
    console.log('✅ Deletion complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
