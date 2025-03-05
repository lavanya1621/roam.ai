import os
import sys
import streamlit as st
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict

# LangGraph and LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# API Keys
import os
import streamlit as st

# Modify your API key retrieval
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
FOURSQUARE_API_KEY = st.secrets.get("FOURSQUARE_API_KEY") or os.getenv("FOURSQUARE_API_KEY")

# Add validation
if not OPENAI_API_KEY or not FOURSQUARE_API_KEY:
    st.error("Please configure API keys in Streamlit secrets or .env file")
    st.stop()
    
# Define the state for our graph
class AgentState(TypedDict):
    city: str
    mood: str
    dietary_preference: str
    activities: List[str]
    restaurants: List[str]
    error: str

class CityExperiencePlanner:
    def __init__(self):
        # Initialize language models with slightly higher creativity
        self.location_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.restaurant_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    def _get_foursquare_venues(self, city: str, categories: List[str], limit: int = 10) -> List[Dict]:
        """Fetch venues from Foursquare API"""
        headers = {
            "Accept": "application/json",
            "Authorization": FOURSQUARE_API_KEY
        }
        
        venues = []
        for category_id in categories:
            params = {
                "query": city,
                "categories": category_id,
                "sort": "POPULARITY",
                "limit": limit
            }
            
            try:
                response = requests.get(
                    "https://api.foursquare.com/v3/places/search", 
                    headers=headers, 
                    params=params
                )
                data = response.json()
                
                for venue in data.get('results', []):
                    venues.append({
                        'name': venue.get('name', 'Unknown Location'),
                        'address': venue.get('location', {}).get('formatted_address', 'No address'),
                    })
            except Exception as e:
                print(f"Foursquare API error: {e}")
        
        return venues

    def location_research_agent(self, state: AgentState):
        """Enhanced Agent to research potential locations based on mood"""
        # Comprehensive mood-based location categories
        mood_categories = {
            "Adventurous": ["16032", "19014", "15029", "19005", "16036", "19011", "15039"],
            "Relaxed": ["13237", "10000", "15026"],
            "Cultural": ["10000", "15026", "16000", "12135"],
            "Romantic": ["13237", "15026", "10026"],
            "Energetic": ["15029", "16032", "13211"],
            "Fun": ["13211", "15029", "16032", "10026", "13306", "16033"]
        }
        
        # Get venues matching mood
        categories = mood_categories.get(state['mood'], [])
        venues = self._get_foursquare_venues(state['city'], categories)
        
        # Prepare venues for LLM processing
        venues_str = "\n".join([f"{v['name']} - {v['address']}" for v in venues[:10]])
        
        # Enhanced Prompt for Consistent Activity Generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate 3 exciting activities for a {mood} experience in {city}.

            STRICT OUTPUT REQUIREMENTS:
            - Provide EXACTLY 3 complete activity descriptions
            - ONLY include the full activity name first and then its description
            - Each activity must have a 2-3 lines about the activity
            - Descriptions should be engaging and capture the essence of {mood}
            - Focus on unique, local experiences in {city}
            - Avoid using any additional headers or formatting
            - Do NOT use Location: or Excitement: headers

            Available Local Venues:
            {venues}"""),
            ("human", "Create 3 compelling activities for an unforgettable day")
        ])
        
        # Create chain and invoke
        chain = prompt | self.location_model | StrOutputParser()
        try:
            result = chain.invoke({
                "city": state['city'],
                "mood": state['mood'],
                "venues": venues_str
            })
            
            # Carefully parse activities
            activities = [act.strip() for act in result.split('\n\n') if act.strip()]
            
            # Ensure exactly 3 activities
            if len(activities) < 3:
                raise ValueError("Not enough activities generated")
            
            return {"activities": activities[:3]}
        
        except Exception as e:
            # Fallback activities if generation fails
            fallback_activities = [
                f"City Tour: Explore the highlights of {state['city']} with a comprehensive walking tour that takes you through the most iconic streets, historical landmarks, and vibrant neighborhoods, offering an immersive experience of the city's rich culture and history.",
                f"Local Market Experience: Dive into the bustling markets of {state['city']}, where you'll weave through colorful stalls, interact with local vendors, sample street food, and discover unique handicrafts that showcase the city's authentic local flavor.",
                f"Cultural Landmark Visit: Embark on a deep dive into {state['city']}'s most significant historical and cultural sites, exploring museums, ancient temples, or architectural marvels that tell the story of the city's fascinating heritage."
            ]
            return {"activities": fallback_activities}

    def restaurant_recommendation_agent(self, state: AgentState):
        """Agent to recommend restaurants based on dietary preference"""
        # Dietary preference categories
        diet_categories = {
            "Vegetarian": ["13376", "13364"],
            "Non-Vegetarian": ["13364"]
        }
        
        # Get restaurants
        categories = diet_categories.get(state['dietary_preference'], [])
        restaurants = self._get_foursquare_venues(state['city'], categories)
        
        # Prepare restaurants for LLM processing
        restaurants_str = "\n".join([f"{r['name']} - {r['address']}" for r in restaurants[:10]])
        
        # Simplified Prompt for Restaurant Recommendations
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Recommend 3 restaurants in {city} with great {dietary_preference} options.

            Available Restaurants:
            {restaurants}

            Output Guidelines:
            - Select 3 restaurants with excellent meal variety
            - If {dietary_preference} is Vegetarian:
              * Look for restaurants with robust vegetarian menu
              * Can include places with both veg and non-veg options
            - Format each recommendation as:
              Restaurant Name and a 2-3 liner on why it's recommended
            - Focus on dining experience and food quality"""),
            ("human", "Give me 3 great restaurant recommendations")
        ])
        
        # Create chain and invoke
        chain = prompt | self.restaurant_model | StrOutputParser()
        restaurants = chain.invoke({
            "city": state['city'],
            "dietary_preference": state['dietary_preference'],
            "restaurants": restaurants_str
        }).split('\n')
        
        # Ensure we return exactly 3 restaurants, cleaning up any empty lines
        return {"restaurants": [rest.strip() for rest in restaurants if rest.strip()][:3]}

    def build_workflow(self):
        """Construct the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("location_research", self.location_research_agent)
        workflow.add_node("restaurant_recommendation", self.restaurant_recommendation_agent)
        
        # Define edges
        workflow.add_edge("location_research", "restaurant_recommendation")
        
        # Set entry and end points
        workflow.set_entry_point("location_research")
        workflow.set_finish_point("restaurant_recommendation")
        
        # Compile the graph
        return workflow.compile()

def main():
    st.title("Roam.ai - Your Personal Day Planner")
    
    # Input validation
    if not OPENAI_API_KEY or not FOURSQUARE_API_KEY:
        st.error("Please set up your API keys in the .env file")
        return
    
    # User inputs
    city = st.text_input("Where are you?")
    
    mood_options = ["Adventurous", "Relaxed", "Cultural", "Romantic", "Energetic", "Fun"]
    mood = st.selectbox("What's your mood?", mood_options)
    
    dietary_options = ["Vegetarian", "Non-Vegetarian"]
    dietary_preference = st.selectbox("Food Preference?", dietary_options)
    
    if st.button("Plan My Day"):
        if city and mood and dietary_preference:
            try:
                # Create planner and workflow
                planner = CityExperiencePlanner()
                workflow = planner.build_workflow()
                
                # Initial state
                initial_state = {
                    "city": city,
                    "mood": mood,
                    "dietary_preference": dietary_preference,
                    "activities": [],
                    "restaurants": [],
                    "error": ""
                }
                
                # Execute workflow
                with st.spinner('Crafting your perfect day...'):
                    result = workflow.invoke(initial_state)
                
                # Display results
                st.subheader(f"🎉 {mood} Activities in {city}")
                for activity in result.get('activities', []):
                    st.markdown(f"- {activity}")
                
                st.subheader(f"🍽️ {dietary_preference} Dining Experiences")
                for restaurant in result.get('restaurants', []):
                    st.markdown(f"- {restaurant}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(f"Error details: {sys.exc_info()}")
        else:
            st.warning("Please fill in all details")

if __name__ == "__main__":
    main()