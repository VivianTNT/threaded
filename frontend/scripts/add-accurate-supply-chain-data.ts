import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function addAccurateSupplyChainData() {
  console.log('Adding accurate supply chain nodes and flows...')

  // Add accurate nodes
  const nodes = [
    // LITHIUM OPERATIONS
    {
      name: 'Greenbushes Lithium Mine',
      node_type: 'mine',
      location: 'Greenbushes, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 210000,
      capacity_unit: 'tonnes',
      parent_company: 'Talison Lithium (Tianqi/IGO/Albemarle JV)',
      description: "World's largest hard rock lithium mine by reserves and production",
      latitude: -33.8333,
      longitude: 116.0667
    },
    {
      name: 'Pilgangoora Lithium Project',
      node_type: 'mine',
      location: 'Pilbara, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 1000000,
      capacity_unit: 'tonnes',
      parent_company: 'Pilbara Minerals',
      description: 'Major spodumene producer with 1 Mtpa capacity after P1000 expansion',
      latitude: -21.1167,
      longitude: 118.5833
    },
    {
      name: 'Mount Marion Lithium Mine',
      node_type: 'mine',
      location: 'Mount Marion, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 600000,
      capacity_unit: 'tonnes',
      parent_company: 'Mineral Resources & Ganfeng JV',
      description: 'High-grade spodumene concentrate production',
      latitude: -30.2167,
      longitude: 119.9833
    },
    {
      name: 'Wodgina Lithium Mine',
      node_type: 'mine',
      location: 'Pilbara, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 750000,
      capacity_unit: 'tonnes',
      parent_company: 'MARBL JV (MinRes/Albemarle)',
      description: 'Restarted operations producing spodumene concentrate',
      latitude: -21.1833,
      longitude: 118.7000
    },
    {
      name: 'Salar de Atacama (Albemarle)',
      node_type: 'mine',
      location: 'Antofagasta Region',
      country: 'Chile',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 52200,
      capacity_unit: 'tonnes',
      parent_company: 'Albemarle Corporation',
      description: "Major lithium brine operation in Chile's Atacama Desert",
      latitude: -23.4500,
      longitude: -68.2500
    },
    {
      name: 'Salar de Atacama (SQM)',
      node_type: 'mine',
      location: 'Antofagasta Region',
      country: 'Chile',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 180000,
      capacity_unit: 'tonnes',
      parent_company: 'Sociedad Quimica y Minera (SQM)',
      description: 'Large-scale lithium carbonate and hydroxide producer',
      latitude: -23.4500,
      longitude: -68.2500
    },
    {
      name: 'Mount Holland (Earl Grey)',
      node_type: 'mine',
      location: 'Mount Holland, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 50000,
      capacity_unit: 'tonnes',
      parent_company: 'Covalent Lithium (SQM/Wesfarmers JV)',
      description: 'Integrated mine-to-hydroxide project with Kwinana refinery',
      latitude: -32.6167,
      longitude: 119.5333
    },
    {
      name: 'Goulamina Lithium Project',
      node_type: 'mine',
      location: 'Bougouni Region',
      country: 'Mali',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 506000,
      capacity_unit: 'tonnes',
      parent_company: 'Ganfeng Lithium',
      description: 'New lithium spodumene operation brought online December 2024',
      latitude: 11.4167,
      longitude: -7.4833
    },
    // COBALT OPERATIONS
    {
      name: 'Tenke Fungurume Mine',
      node_type: 'mine',
      location: 'Lualaba Province',
      country: 'Democratic Republic of Congo',
      commodities: ['Cobalt', 'Copper'],
      status: 'active',
      capacity: 28500,
      capacity_unit: 'tonnes',
      parent_company: 'CMOC Group (80%)',
      description: 'Largest cobalt producer globally, also produces copper',
      latitude: -10.6000,
      longitude: 26.4000
    },
    {
      name: 'Mutanda Mine',
      node_type: 'mine',
      location: 'Lualaba Province',
      country: 'Democratic Republic of Congo',
      commodities: ['Cobalt', 'Copper'],
      status: 'active',
      capacity: 27000,
      capacity_unit: 'tonnes',
      parent_company: 'Glencore (95%)',
      description: "World's largest cobalt mine by reserves",
      latitude: -10.9167,
      longitude: 27.6167
    },
    {
      name: 'Kamoto Copper Company (KCC)',
      node_type: 'mine',
      location: 'Kolwezi, Katanga',
      country: 'Democratic Republic of Congo',
      commodities: ['Cobalt', 'Copper'],
      status: 'active',
      capacity: 35000,
      capacity_unit: 'tonnes',
      parent_company: 'Glencore (75%) & Gecamines (25%)',
      description: 'Major copper-cobalt operation in DRC copper belt',
      latitude: -10.7167,
      longitude: 25.4667
    },
    // NICKEL OPERATIONS
    {
      name: 'Weda Bay Nickel Project',
      node_type: 'mine',
      location: 'Halmahera Island',
      country: 'Indonesia',
      commodities: ['Nickel', 'Cobalt'],
      status: 'active',
      capacity: 100000,
      capacity_unit: 'tonnes',
      parent_company: 'Tsingshan Holding Group',
      description: "Indonesia's largest nickel laterite project, 52% of top 10 global nickel production",
      latitude: 0.4000,
      longitude: 127.9000
    },
    {
      name: 'Sorowako Mine (Vale Indonesia)',
      node_type: 'mine',
      location: 'South Sulawesi',
      country: 'Indonesia',
      commodities: ['Nickel'],
      status: 'active',
      capacity: 64000,
      capacity_unit: 'tonnes',
      parent_company: 'Vale Indonesia',
      description: 'Integrated lateritic nickel mining and processing operation, 3% of global production',
      latitude: -2.5333,
      longitude: 121.3333
    },
    {
      name: 'PT Halmahera Persada Lygend',
      node_type: 'mine',
      location: 'Obi Island',
      country: 'Indonesia',
      commodities: ['Nickel', 'Cobalt'],
      status: 'active',
      capacity: 95000,
      capacity_unit: 'tonnes',
      parent_company: 'Ningbo Lygend Mining',
      description: 'Designed capacity of 120kt nickel-cobalt compounds per annum',
      latitude: -1.6000,
      longitude: 127.5000
    },
    {
      name: 'Norilsk-Talnakh Mining Complex',
      node_type: 'mine',
      location: 'Norilsk, Krasnoyarsk Krai',
      country: 'Russia',
      commodities: ['Nickel', 'Copper', 'Palladium'],
      status: 'active',
      capacity: 208577,
      capacity_unit: 'tonnes',
      parent_company: 'Nornickel',
      description: "World's largest palladium and high-grade nickel producer",
      latitude: 69.3333,
      longitude: 88.2167
    },
    // COPPER OPERATIONS
    {
      name: 'Escondida Mine',
      node_type: 'mine',
      location: 'Atacama Desert',
      country: 'Chile',
      commodities: ['Copper'],
      status: 'active',
      capacity: 1280000,
      capacity_unit: 'tonnes',
      parent_company: 'BHP (57.5%) & Rio Tinto (30%)',
      description: "World's largest copper mine by production and reserves",
      latitude: -24.3667,
      longitude: -69.0833
    },
    {
      name: 'Grasberg Mine',
      node_type: 'mine',
      location: 'Papua Province',
      country: 'Indonesia',
      commodities: ['Copper', 'Gold'],
      status: 'active',
      capacity: 816466,
      capacity_unit: 'tonnes',
      parent_company: 'Freeport-McMoRan & PT Indonesia',
      description: "One of world's largest copper and gold mines",
      latitude: -4.0533,
      longitude: 137.1167
    },
    {
      name: 'Collahuasi Mine',
      node_type: 'mine',
      location: 'Tarapacá Region',
      country: 'Chile',
      commodities: ['Copper'],
      status: 'active',
      capacity: 558636,
      capacity_unit: 'tonnes',
      parent_company: 'Anglo American (44%) & Glencore (44%)',
      description: 'High-altitude copper mine in northern Chile',
      latitude: -20.9667,
      longitude: -68.6833
    },
    {
      name: 'Kamoa-Kakula Copper Complex',
      node_type: 'mine',
      location: 'Lualaba Province',
      country: 'Democratic Republic of Congo',
      commodities: ['Copper'],
      status: 'active',
      capacity: 450000,
      capacity_unit: 'tonnes',
      parent_company: 'Ivanhoe Mines & Zijin Mining',
      description: "World's third-largest copper mining complex, phase 3 concentrator 5 Mtpa",
      latitude: -10.4667,
      longitude: 25.7500
    },
    // RARE EARTH OPERATIONS
    {
      name: 'Bayan Obo Mine',
      node_type: 'mine',
      location: 'Inner Mongolia',
      country: 'China',
      commodities: ['Rare Earth Elements', 'Iron'],
      status: 'active',
      capacity: 135000,
      capacity_unit: 'tonnes',
      parent_company: 'China Northern Rare Earth (Baogang Group)',
      description: "World's largest REE mine, 40% of global reserves, nearly 50% of production",
      latitude: 41.7667,
      longitude: 109.9667
    },
    {
      name: 'Mountain Pass Mine',
      node_type: 'mine',
      location: 'California',
      country: 'United States',
      commodities: ['Rare Earth Elements'],
      status: 'active',
      capacity: 45000,
      capacity_unit: 'tonnes',
      parent_company: 'MP Materials',
      description: 'Only active REE mine in USA, major NdPr producer',
      latitude: 35.4833,
      longitude: -115.5333
    },
    {
      name: 'Mount Weld Mine',
      node_type: 'mine',
      location: 'Western Australia',
      country: 'Australia',
      commodities: ['Rare Earth Elements'],
      status: 'active',
      capacity: 13000,
      capacity_unit: 'tonnes',
      parent_company: 'Lynas Rare Earths',
      description: 'High-grade NdPr mine, expanding to 12kt NdPr annually by 2025',
      latitude: -28.8500,
      longitude: 122.4000
    },
    // PROCESSING & REFINING
    {
      name: 'Kwinana Lithium Refinery',
      node_type: 'refinery',
      location: 'Kwinana, Western Australia',
      country: 'Australia',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 50000,
      capacity_unit: 'tonnes',
      parent_company: 'Covalent Lithium (SQM/Wesfarmers)',
      description: 'Lithium hydroxide refinery targeting 50ktpa capacity by 2026',
      latitude: -32.2167,
      longitude: 115.7833
    },
    {
      name: 'Freeport Smelting Works',
      node_type: 'smelter',
      location: 'Gresik, East Java',
      country: 'Indonesia',
      commodities: ['Copper'],
      status: 'active',
      capacity: 300000,
      capacity_unit: 'tonnes',
      parent_company: 'PT Freeport Indonesia',
      description: 'Copper smelting and refining complex processing Grasberg concentrate',
      latitude: -7.1667,
      longitude: 112.6500
    },
    {
      name: 'Jinchuan Nickel Smelter',
      node_type: 'smelter',
      location: 'Jinchang, Gansu',
      country: 'China',
      commodities: ['Nickel', 'Cobalt'],
      status: 'active',
      capacity: 140000,
      capacity_unit: 'tonnes',
      parent_company: 'Jinchuan Group',
      description: "China's largest nickel producer and third-largest cobalt producer globally",
      latitude: 38.5000,
      longitude: 102.1833
    },
    // BATTERY MANUFACTURERS
    {
      name: 'CATL Ningde Plant',
      node_type: 'manufacturer',
      location: 'Ningde, Fujian',
      country: 'China',
      commodities: ['Lithium'],
      status: 'active',
      capacity: 500000,
      capacity_unit: 'MWh',
      parent_company: 'Contemporary Amperex Technology (CATL)',
      description: "World's largest EV battery manufacturer, 37% global market share 2024",
      latitude: 26.6667,
      longitude: 119.5500
    },
    {
      name: 'LG Energy Solution Ochang',
      node_type: 'manufacturer',
      location: 'Ochang, North Chungcheong',
      country: 'South Korea',
      commodities: ['Lithium', 'Nickel', 'Cobalt'],
      status: 'active',
      capacity: 200000,
      capacity_unit: 'MWh',
      parent_company: 'LG Energy Solution',
      description: 'Major battery cell manufacturing facility serving global EV market',
      latitude: 36.7167,
      longitude: 127.4833
    },
    {
      name: 'Panasonic Osaka Factory',
      node_type: 'manufacturer',
      location: 'Osaka',
      country: 'Japan',
      commodities: ['Lithium', 'Nickel', 'Cobalt'],
      status: 'active',
      capacity: 150000,
      capacity_unit: 'MWh',
      parent_company: 'Panasonic Corporation',
      description: 'Tesla Gigafactory partner and major cylindrical cell producer',
      latitude: 34.6833,
      longitude: 135.5167
    }
  ]

  const { data: insertedNodes, error: nodesError } = await supabase
    .from('supply_chain_nodes')
    .insert(nodes)
    .select()

  if (nodesError) {
    console.error('Error inserting nodes:', nodesError)
    throw nodesError
  }

  console.log(`Successfully inserted ${insertedNodes.length} nodes`)

  // Now add the flows connecting these nodes
  const flowMappings = [
    { source: 'Greenbushes Lithium Mine', target: 'Kwinana Lithium Refinery', commodity: 'Lithium', volume: 45000 },
    { source: 'Pilgangoora Lithium Project', target: 'CATL Ningde Plant', commodity: 'Lithium', volume: 180000 },
    { source: 'Grasberg Mine', target: 'Freeport Smelting Works', commodity: 'Copper', volume: 750000 },
    { source: 'Weda Bay Nickel Project', target: 'LG Energy Solution Ochang', commodity: 'Nickel', volume: 85000 },
    { source: 'Tenke Fungurume Mine', target: 'Jinchuan Nickel Smelter', commodity: 'Cobalt', volume: 25000 }
  ]

  for (const mapping of flowMappings) {
    const { data: sourceNode } = await supabase
      .from('supply_chain_nodes')
      .select('id')
      .eq('name', mapping.source)
      .single()

    const { data: targetNode } = await supabase
      .from('supply_chain_nodes')
      .select('id')
      .eq('name', mapping.target)
      .single()

    if (sourceNode && targetNode) {
      await supabase.from('supply_chain_flows').insert({
        source_node_id: sourceNode.id,
        target_node_id: targetNode.id,
        commodity: mapping.commodity,
        volume: mapping.volume,
        volume_unit: 'tonnes',
        status: 'active',
        transportation_mode: ['Ship'],
        avg_transit_time_days: 30
      })
    }
  }

  console.log('Successfully added accurate supply chain data!')
}

addAccurateSupplyChainData()
  .then(() => {
    console.log('Done!')
    process.exit(0)
  })
  .catch((error) => {
    console.error('Error:', error)
    process.exit(1)
  })
