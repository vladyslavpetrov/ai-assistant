from router import Router

if __name__ == "__main__":
    router = Router()
    user_input = input("Enter your query: ")
    response = router.run_agent(user_input)
    print("Response:", response)