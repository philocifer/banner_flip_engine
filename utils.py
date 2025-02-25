import csv
import json
import asyncio
import time

def csv_to_json(csv_file_path, json_file_path):
    """Convert CSV store data to cleaned JSON format"""
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = []
            
            for row in csv_reader:
                # Clean empty values and strip whitespace
                cleaned_row = {
                    key: value.strip() 
                    for key, value in row.items() 
                    if value.strip()
                }
                data.append(cleaned_row)
            
            print(f"Converted {len(data)} rows from CSV")
            
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2)
            print(f"Saved JSON to {json_file_path}")
            
    except Exception as e:
        print(f"CSV conversion failed: {str(e)}")
        raise

async def generate_semantic_content_async(input_json_path, output_txt_path, max_concurrent=50):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    processed = 0
    start_time = time.time()
    lock = asyncio.Lock()
    
    template = """STORE ANALYSIS: {store_name} ({store_id})
Location: {street_address}, {city_name}

Store {store_id} - {store_name} operates as a {store_status} {trade_sub_channel} establishment at {street_address}, {city_name}, {state_iso2_code} {postal_code} (FIPS {state_fips_code}-{county_fips_code}). Geographically precise at coordinates {latitude},{longitude} ({lat_long_precision}), this location generates ${annual_sales:,.0f} in annual sales ({annual_sales_range}) from its {square_feet} square foot space. The operation employs {full_time_employee_count} full-time staff across {checkout_lane_count} checkout lanes, yielding a sales density of ${sales_density:,.2f}/sqft. Owned by {owner_name} (Family ID: {owner_family_id}) as part of a {parent_company_store_count}-location network, the store sources inventory through {supplier_name} (Supplier ID: {supplier_id}, Family ID: {supplier_family_id}) to maintain its position in the {trade_channel} sector's {trade_sub_channel} segment.

""" + "="*80

    sem = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_event_loop()

    async def process_item(item):
        async with sem:
            try:
                # Convert numeric fields safely
                annual_sales = float(str(item.get('annual_sales', '0')).replace(',', ''))
                square_feet = float(str(item.get('square_feet', '1')).replace(',', ''))
                sales_density = annual_sales / square_feet if square_feet else 0

                # Prepare template variables
                template_vars = {
                    'sales_density': sales_density,
                    'annual_sales': annual_sales,
                    'square_feet': square_feet,
                    **{k: item.get(k, 'N/A') for k in [
                        'store_id', 'store_name', 'street_address', 'city_name',
                        'state_iso2_code', 'postal_code', 'state_fips_code',
                        'county_fips_code', 'latitude', 'longitude',
                        'lat_long_precision', 'annual_sales_range',
                        'full_time_employee_count', 'checkout_lane_count',
                        'owner_name', 'owner_family_id', 'parent_company_store_count',
                        'supplier_name', 'supplier_id', 'supplier_family_id',
                        'trade_channel', 'trade_sub_channel', 'store_status'
                    ]}
                }

                result = await loop.run_in_executor(
                    None, 
                    lambda: template.format(**template_vars)
                )
                
                async with lock:
                    nonlocal processed
                    processed += 1
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"\rProcessed {processed}/{total} ({processed/total:.1%}) | {rate:.1f} items/sec", end='')
                return result
                
            except Exception as e:
                print(f"\nError processing item {item.get('store_id')}: {str(e)}")
                return f"/* ERROR PROCESSING STORE {item.get('store_id')} */\n"

    tasks = [process_item(item) for item in data]
    results = await asyncio.gather(*tasks)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"\nCompleted processing {total} items in {time.time()-start_time:.1f} seconds")

# Updated sync wrapper
def generate_semantic_content(*args, **kwargs):
    return asyncio.run(generate_semantic_content_async(*args, **kwargs))

if __name__ == "__main__":
    csv_to_json('data/competitor_store_200.csv', 'data/competitor_store_200.json')
    generate_semantic_content('data/competitor_store_200.json', 'data/enhanced_store_data_200.txt', max_concurrent=50)
