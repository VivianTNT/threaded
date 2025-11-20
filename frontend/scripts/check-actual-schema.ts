/**
 * Check the actual database schema for projects table
 */

import { Pool } from 'pg';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const password = process.env.DATABASE_PASSWORD || '';

const projectRef = supabaseUrl.replace('https://', '').split('.')[0];
const connectionString = `postgresql://postgres:${password}@db.${projectRef}.supabase.co:5432/postgres`;

const pool = new Pool({
  connectionString,
  ssl: { rejectUnauthorized: false }
});

async function checkSchema() {
  const client = await pool.connect();

  try {
    console.log('🔍 Checking actual projects table schema...\n');

    const result = await client.query(`
      SELECT
        column_name,
        data_type,
        character_maximum_length,
        is_nullable,
        column_default
      FROM information_schema.columns
      WHERE table_name = 'projects'
      ORDER BY ordinal_position;
    `);

    console.log(`✓ Found ${result.rows.length} columns in projects table:\n`);

    result.rows.forEach((row, index) => {
      const nullable = row.is_nullable === 'YES' ? 'NULL' : 'NOT NULL';
      const type = row.character_maximum_length
        ? `${row.data_type}(${row.character_maximum_length})`
        : row.data_type;
      const defaultVal = row.column_default ? ` DEFAULT ${row.column_default}` : '';

      console.log(`${String(index + 1).padStart(2)}. ${row.column_name.padEnd(35)} ${type.padEnd(25)} ${nullable}${defaultVal}`);
    });

    // Check specifically for financial columns
    console.log('\n💰 Financial-related columns:');
    const financialCols = result.rows.filter(row =>
      row.column_name.includes('npv') ||
      row.column_name.includes('irr') ||
      row.column_name.includes('capex') ||
      row.column_name.includes('discount') ||
      row.column_name.includes('financial')
    );

    if (financialCols.length > 0) {
      financialCols.forEach(col => {
        console.log(`  - ${col.column_name} (${col.data_type})`);
      });
    } else {
      console.log('  (none found)');
    }

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    throw error;
  } finally {
    client.release();
    await pool.end();
  }
}

checkSchema()
  .then(() => process.exit(0))
  .catch(error => {
    console.error(error);
    process.exit(1);
  });
