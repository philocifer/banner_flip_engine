import requests
import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain_core.tools import tool
from dotenv import load_dotenv
from json import JSONDecodeError

load_dotenv()

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    raise ValueError("CENSUS_API_KEY is not set")

class StoreDemographicProfile:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.census.gov/data/2021/acs/acs5"
    
    def get_tract_code(self, lat, lon):
        response = requests.get(
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
            params={
                "x": lon,
                "y": lat,
                "benchmark": "Public_AR_Current",
                "vintage": "ACS2021_Current",
                "format": "json"
            }
        )
        if response.status_code != 200:
            raise ValueError(f"Geocoding failed: {response.text}")
        
        try:
            tract_data = response.json()
        except JSONDecodeError:
            raise ValueError("Invalid JSON response from geocoding service")
        
        if 'geographies' not in tract_data.get('result', {}):
            raise ValueError("No geographic data found for coordinates")
        
        tracts = tract_data['result']['geographies'].get('Census Tracts', [])
        if not tracts:
            raise ValueError("No census tract found for given coordinates")
        
        return {
            "state": tracts[0]['STATE'],
            "county": tracts[0]['COUNTY'],
            "tract": tracts[0]['TRACT'],
            "land_area": float(tracts[0]['AREALAND'])  # Square meters
        }

    def _get_census_data(self, variables, state_fips, county_fips, tract):
        """Helper to fetch census data for specific variables"""
        response = requests.get(
            self.base_url,
            params={
                "get": ",".join(variables),
                "for": f"tract:{tract}",
                "in": f"state:{state_fips} county:{county_fips}",
                "key": self.api_key
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Census API error: {response.text}")
        
        try:
            response_data = response.json()
        except JSONDecodeError:
            raise ValueError("Invalid JSON response from Census API")
        
        if len(response_data) < 2:
            raise ValueError("No data available for this census tract")
            
        return response_data[1]  # Return just the data row

    def get_profile(self, state_fips, county_fips, lat, lon):
        geo = self.get_tract_code(lat, lon)
        
        # Group 1: Age/Gender Distribution (30 variables) + Population (1)
        group1 = [
            # Population (1)
            "B01003_001E",  # Total population
            
            # Age/Gender Distribution (30 variables)
            # Male 0-17
            "B01001_003E", "B01001_004E", "B01001_005E",
            # Female 0-17
            "B01001_027E", "B01001_028E", "B01001_029E",
            # Male 18-34
            "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
            "B01001_011E", "B01001_012E", "B01001_013E",
            # Female 18-34
            "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",
            "B01001_035E", "B01001_036E", "B01001_037E",
            # Male 35-64
            "B01001_014E", "B01001_015E", "B01001_016E",
            "B01001_017E", "B01001_018E", "B01001_019E",
            # Female 35-64
            "B01001_038E", "B01001_039E", "B01001_040E",
            "B01001_041E", "B01001_042E", "B01001_043E"
        ]
        
        # Group 2: Remaining Demographics (22 variables)
        group2 = [
            # Economic Indicators
            "B19013_001E",  # Median income
            "B23025_003E",  # Labor force
            "B23025_005E",  # Unemployment
            
            # Education
            "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
            
            # Family Structure
            "B11003_001E",  # Families
            "B25010_001E",  # Household size
            "B11001_002E",  # Married couples
            
            # Ethnicity
            "B02001_002E", "B02001_003E", "B02001_004E", 
            "B02001_005E", "B02001_006E", "B03002_012E",  # Hispanic
            
            # Housing
            "B25003_001E", "B25003_002E", "B25024_001E",
            
            # Economic Inequality
            "B19083_001E"   # Gini index
        ]

        # Get data for each logical group
        data_group1 = self._get_census_data(group1, state_fips, county_fips, geo['tract'])
        data_group2 = self._get_census_data(group2, state_fips, county_fips, geo['tract'])
        
        # Combine while maintaining group order
        combined_data = data_group1 + data_group2
        
        return self._process_data(combined_data, geo)

    def _process_data(self, data, geo):
        # Helper function to sum multiple columns with empty check
        def sum_columns(indices):
            return sum(float(data[i]) if data[i] else 0.0 for i in indices)
        
        # Safely parse total population with fallback
        total_pop = float(data[0]) if data[0] else 0.0
        if total_pop <= 0:
            return {"error": "No population data available for this location"}
        
        land_area_sq_miles = geo['land_area'] / 2589988  # Convert to square miles
        
        # Calculate gender distribution with safe access
        male_pop = sum_columns([1,2,3,7,8,9,10,11,12,13,21,22,23,24,25,26])
        female_pop = sum_columns([4,5,6,14,15,16,17,18,19,20,27,28,29,30,31,32])

        # Safe value parsing with fallbacks
        def safe_parse(index, default=0.0):
            try:
                return float(data[index]) if (index < len(data) and data[index]) else default
            except (ValueError, TypeError):
                return default

        return {
            # Population Density
            "population_density": f"{total_pop/land_area_sq_miles:.1f} per sq mile" if land_area_sq_miles > 0 else "N/A",
            
            # Gender Distribution
            "gender_distribution": {
                "male": (male_pop/total_pop * 100) if total_pop > 0 else 0.0,
                "female": (female_pop/total_pop * 100) if total_pop > 0 else 0.0
            },
            
            # Age Distribution with safe indices
            "age_distribution": {
                "0-17": (sum_columns([1,2,3,4,5,6])/total_pop * 100) if total_pop > 0 else 0.0,
                "18-34": (sum_columns([7,8,9,10,11,12,13,14])/total_pop * 100) if total_pop > 0 else 0.0,
                "35-49": (sum_columns([15,16,17,18,19,20])/total_pop * 100) if total_pop > 0 else 0.0,
                "50-64": (sum_columns([21,22,23,24,25,26])/total_pop * 100) if total_pop > 0 else 0.0,
                "65+": (sum_columns([27,28,29,30,31,32])/total_pop * 100) if total_pop > 0 else 0.0
            },
            
            # Economic Indicators with safe access
            "median_income": safe_parse(33),
            "labor_force_participation": safe_parse(34)/100,
            "unemployment_rate": safe_parse(35)/100,
            
            # Education with division guard
            "college_plus": (sum_columns([36,37,38,39])/safe_parse(40, 1.0)) * 100 if safe_parse(40, 0.0) > 0 else 0.0,
            
            # Family Structure with safe division
            "average_household_size": safe_parse(42),
            "married_couple_pct": (safe_parse(43)/safe_parse(41, 1.0)) * 100 if safe_parse(41, 0.0) > 0 else 0.0,
            
            # Ethnicity with population ratio safety
            "ethnic_breakdown": {
                "white": (safe_parse(44)/total_pop * 100) if total_pop > 0 else 0.0,
                "african_american": (safe_parse(45)/total_pop * 100) if total_pop > 0 else 0.0,
                "native_american": (safe_parse(46)/total_pop * 100) if total_pop > 0 else 0.0,
                "asian": (safe_parse(47)/total_pop * 100) if total_pop > 0 else 0.0,
                "pacific_islander": (safe_parse(48)/total_pop * 100) if total_pop > 0 else 0.0,
                "hispanic": (safe_parse(49)/total_pop * 100) if total_pop > 0 else 0.0
            },
            
            # Housing with division guard
            "homeownership_rate": (safe_parse(50)/safe_parse(49, 1.0)) * 100 if safe_parse(49, 0.0) > 0 else 0.0,
            "income_inequality": safe_parse(53)
        }

def load_agent():
    class State(TypedDict):
        state_fips: str
        county_fips: str
        lat: float
        lon: float
        profile_data: dict
        profile_report: str

    def retrieve_profile(state):
        profiler = StoreDemographicProfile(CENSUS_API_KEY)
        profile_data = profiler.get_profile(state['state_fips'], state['county_fips'], state['lat'], state['lon'])
        return {"profile_data": profile_data}

    PROFILE_PROMPT = """\
    Transform this raw demographic data into a retail grocery store location profile. Use this structure:

    **Store Location Profile: {location_coordinates}**

    1. **Market Potential Analysis**
    - Population Density: {population_density} ({context})
    - Gender Distribution:
        * Male: {gender_male:.1f}%
        * Female: {gender_female:.1f}%
    - Age Distribution: 
        * Children (0-17): {age_0_17:.1f}%
        * Young Adults (18-34): {age_18_34:.1f}%
        * Families (35-49): {age_35_49:.1f}%
        * Older Adults (50+): {age_65_plus:.1f}%
    - Household Economics: Median income ${median_income:,.0f}, {labor_force_participation:.1%} workforce participation
    - Grocery Spend: {grocery_spend}

    2. **Cultural & Lifestyle Factors**
    - Education: {college_plus:.1f}% college graduates
    - Family Structure: Average {average_household_size:.1f} persons/household, {married_couple_pct:.1f}% married couples
    - Cultural Diversity:
        * White: {ethnic_white:.1f}%
        * African American: {ethnic_african_american:.1f}%
        * Native American: {ethnic_native_american:.1f}%
        * Asian: {ethnic_asian:.1f}%
        * Pacific Islander: {ethnic_pacific:.1f}%
        * Hispanic/Latino: {ethnic_hispanic:.1f}%

    3. **Location Strategy Recommendations**
    - Housing Insights: {homeownership_rate:.1f}% homeownership, {housing_type} predominant
    - Income Inequality: Gini index {income_inequality:.2f} ({interpretation})
    - Suggested Positioning: {positioning} based on income and competition

    4. **Key Opportunities**
    - 3 bullet points highlighting favorable demographic factors
    - Potential product mix suggestions based on age/ethnicity

    5. **Data Quality Notes**
    {data_quality}

    Formatting Rules:
    - Use clean markdown-free text
    - Convert all percentages to 1 decimal place
    - Add contextual interpretations in parentheses
    - Highlight unusual data points (>2 standard deviations from regional averages)
    - Keep analysis under 400 words
    """

    profile_prompt = ChatPromptTemplate.from_template(PROFILE_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_profile_report(state):
        # Extract and format the required data
        profile_data = state['profile_data']
        ethnic = profile_data.get("ethnic_breakdown", {})
        
        # Calculate grocery spend with safety checks
        try:
            grocery_spend = (profile_data["median_income"] * 0.07) / (1 + profile_data["income_inequality"])
        except (ZeroDivisionError, KeyError, TypeError):
            grocery_spend = 0.0

        formatted_data = {
            "location_coordinates": f"{state['lat']},{state['lon']}",
            "population_density": profile_data.get("population_density", "N/A"),
            "gender_male": profile_data.get("gender_distribution", {}).get("male", 0.0),
            "gender_female": profile_data.get("gender_distribution", {}).get("female", 0.0),
            "age_0_17": profile_data.get("age_distribution", {}).get("0-17", 0.0),
            "age_18_34": profile_data.get("age_distribution", {}).get("18-34", 0.0),
            "age_35_49": profile_data.get("age_distribution", {}).get("35-49", 0.0),
            "age_65_plus": profile_data.get("age_distribution", {}).get("65+", 0.0),
            "median_income": profile_data.get("median_income", 0),
            "labor_force_participation": profile_data.get("labor_force_participation", 0),
            "unemployment_rate": profile_data.get("unemployment_rate", 0),
            "income_inequality": profile_data.get("income_inequality", 0),
            "college_plus": profile_data.get("college_plus", 0),
            "average_household_size": profile_data.get("average_household_size", 0),
            "married_couple_pct": profile_data.get("married_couple_pct", 0),
            "ethnic_white": ethnic.get("white", 0.0),
            "ethnic_african_american": ethnic.get("african_american", 0.0),
            "ethnic_native_american": ethnic.get("native_american", 0.0),
            "ethnic_asian": ethnic.get("asian", 0.0),
            "ethnic_pacific": ethnic.get("pacific_islander", 0.0),
            "ethnic_hispanic": ethnic.get("hispanic", 0.0),
            "homeownership_rate": profile_data.get("homeownership_rate", 0),
            "grocery_spend": f"{grocery_spend:.1f}%"
        }

        # Add calculated fields with defaults
        formatted_data.update({
            "context": "urban" if "per sq mile" in formatted_data["population_density"] and 
                            float(formatted_data["population_density"].split()[0]) > 1000 else "rural",
            "housing_type": "single-family" if formatted_data["homeownership_rate"] > 50 else "multi-unit",
            "interpretation": "high inequality" if formatted_data["income_inequality"] > 0.4 else "moderate",
            "positioning": "premium" if formatted_data["median_income"] > 75000 else "value"
        })

        # Data quality analysis
        data_notes = []
        
        # Check gender distribution total
        gender_total = formatted_data["gender_male"] + formatted_data["gender_female"]
        if not (99.5 <= gender_total <= 100.5):
            data_notes.append(f"Gender distribution totals {gender_total:.1f}% (expected ~100%) - difference may include non-binary categories")
            
        # Check grocery spend validity
        if grocery_spend <= 0:
            data_notes.append("Grocery spend calculation unavailable - missing income or inequality data")
            
        # Check population density
        if "N/A" in formatted_data["population_density"]:
            data_notes.append("Population density unavailable - missing land area data")
            
        # Check age distribution total
        age_total = sum([
            formatted_data["age_0_17"],
            formatted_data["age_18_34"],
            formatted_data["age_35_49"],
            formatted_data["age_65_plus"]
        ])
        if not (95 <= age_total <= 105):  # Allow wider margin for age grouping approximations
            data_notes.append(f"Age distribution totals {age_total:.1f}% - categories may exclude some age groups")

        # Add data quality section
        if data_notes:
            formatted_data["data_quality"] = "Data Notes:\n- " + "\n- ".join(data_notes)
        else:
            formatted_data["data_quality"] = "All data appears complete and consistent"

        messages = profile_prompt.format_messages(**formatted_data)
        response = llm.invoke(messages)
        return {"profile_report": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve_profile, generate_profile_report])
    graph_builder.add_edge(START, "retrieve_profile")
    graph = graph_builder.compile()
    return graph

profile_agent = load_agent()

@tool
def profile_agent_tool(state_fips: str, county_fips: str, lat: float, lon: float) -> str:
    """Useful for when you need to generate or analyze a retail grocery store location profile. 
    Use other tools to get the location coordinates.
    """
    response = profile_agent.invoke({"state_fips": state_fips, "county_fips": county_fips, "lat": lat, "lon": lon})
    return response["profile_report"]

if __name__ == "__main__":
    result = profile_agent_tool.invoke({
        "state_fips": "48",
        "county_fips": "215".zfill(3),  # Ensure 3-digit format
        "lat": 26.1045,
        "lon": -97.9523
    })
    print(result)