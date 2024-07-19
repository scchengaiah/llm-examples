from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from dotenv import load_dotenv

environment_loaded = load_dotenv("./.env")

print("Environment Loaded: {environment_loaded}")

## Setup Tools
serper_internet_search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

## Setup Agents
# Agent 1: destination_researcher
destination_researcher_agent = Agent(
    role="Senior Destination Researcher",
    goal="Make sure to find gorgeous and attractive "
         "destination based on the customer preferences.",
    tools = [scrape_tool, serper_internet_search_tool],
    verbose=True,
    backstory=(
        "As a Senior Destination Researcher, "
        "you have the ability to find the best vacation destinations "
        "based on the customer preferences. "
        "You have access to the internet to search for the best destinations. "
        "You have the responsibility to adhere to the customer preferences. "
        "Your choice of destinations should result in customer delight."
    )
)

# Agent 2: travel_planner
travel_planner_agent = Agent(
    role="Senior Budget Travel Expert",
    goal="To provide affordable staying and travel options for the identified destination(s).",
    tools = [scrape_tool, serper_internet_search_tool],
    verbose=True,
    backstory=(
        "As a Senior Budget Travel Expert, "
        "You have the ability to find the best budget-friendly transport options "
        "based on the researched destinations."
        "You have an exceptional ability to identify the budget-friendly accommodation options "
        "for the identified destination(s). "
        "You have access to the internet to search for the best travel and accomodation options. "
        "Ensure that the accommodation option for the different destination(s) is provided."
        "You shall be rewarded with attractive incentives if you can offer the customer "
        "the best travel and accommodation options."

    )
)

# Agent 3:
vacation_itinerary_specialist_agent = Agent(
    role="Vacation Itinerary Specialist",
    goal="Provide an elaborative itinerary for the identified destination(s).",
    tools = [scrape_tool, serper_internet_search_tool],
    verbose=True,
    backstory=(
        "Based on the identified destination(s), accommodation and travel options, "
        "you are responsible to provide an elaborative and intuitive itinerary for the customer."
        "Ensure that budget friendly accommodation options are provided for the identified destination(s)."
        "Ensure the provided itinerary is as detailed as possible which should include the tenative travel dates and timeline recommendation, "
        "accommodation information and travel option possibilities between different destinations. "
    )
)

## Setup Tasks
# Task 1: destination_researcher
destination_researcher_task = Task(
    description=(
        "Conduct an in-depth research on the tourist attractions for the country {destination_country}. "
        "The identified destination(s) should align with the customer preferences."
        "Here are the customer preferences: {travel_preferences}. "
        "The outcome of this research is crucial for the travel planning process. "
    ),
    expected_output=(
        "Attractive destination(s) that align with the customer "
        "preferences for the country {destination_country}. "
    ),
    human_input = True,
    agent=destination_researcher_agent
)


# Task 2: travel_planner
travel_planner_task = Task(
    description=(
        "Based on the research on the destination(s) for the country {destination_country}, "
        "provide a budget-friendly travel and acommodation options for the identified "
        "destination(s). "
        "Here are the travel details from the customer:\n"
        "Travelling from: {travelling_from}\n"
        "Travel dates/Month: {travel_dates}\n"
        "Travel duration: {travel_duration}\n"
        "Passenger details: {passenger_details}\n"
        "Accomodation Location Preferences: {accomodation_location_preferences}\n"
        "Bugdet: {budget}\n"
        "Try to offer the customer the best travel and accommodation options. "
        "Budget can be slightly exceeded if necessary to provide the customer "
        "with the best travel and accommodation options. "
    ),
    expected_output=(
        "A comprehensive report on budget-friendly travel and accommodation options for the identified destination(s) and based on the travel details from the customer."
        "Always strive to provie the best possible travel and accommodation options."
    ),
    human_input = True,
    context = [destination_researcher_task],
    agent=travel_planner_agent
)

# Task 3: vacation_itinerary_specialist
vacation_itinerary_specialist_task = Task(
    description=(
        "Based on the research on the destination(s) for the country {destination_country}, and "
        "the travel and accommodation options, provide an elaborative itinerary."
        "Ensure the itinerary is complete and as detailed as possible. "
    ),
    expected_output=(
        "A comprehensive itinerary report for the identified destination(s) in markdown format. "
        "The itinerary should be complete and as detailed as possible. "
        "Provide the customer with the accommodation and travel options between different destinations."
        "Ensure to drill down the itinerary providing recommendations of time at which the customer "
        "should leave or arrive between different destinations."
        "The quality of itinerary should result in customer delight and contain all necessary "
        "details for a memorable vacation."
    ),
    context = [destination_researcher_task, travel_planner_task],
    human_input = True,
    agent=vacation_itinerary_specialist_agent
)

# Initialize Crew
vacation_planner_crew = Crew(
    agents=[
        destination_researcher_agent,
        travel_planner_agent,
        vacation_itinerary_specialist_agent
    ],
    tasks=[destination_researcher_task, travel_planner_task, vacation_itinerary_specialist_task],
    verbose=True
)

inputs = {
    "destination_country": "Switzerland",
    "travel_preferences": "Gorgeous and nature sight seeings. Attractive destinations that offers unforgettable travel experiences.",
    "travelling_from": "Nuremberg",
    "travel_dates": "May 2025",
    "travel_duration": "4 nights 5 days",
    "passenger_details": "2 adults and 1 child(5 years old)",
    "accomodation_location_preferences": "Close to the city centre or within walking distance to the city centre.",
    "budget": "1700 EUR",
}

result = vacation_planner_crew.kickoff(inputs=inputs)

print(result)