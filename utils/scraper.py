# AER202 Web Scraper Script to collect images from Airliners.net

from bs4 import BeautifulSoup
import requests

# Add the first search result for specific aircraft type here
firstSearch = 'https://www.airliners.net/photo/Air-Canada/Airbus-A321-211/5750199?qsp=eJwtjDsOg0AMRO/imiZAPqILF0iKXMDymmSlDaxsFyDE3bOGdH4zfrMCTaPxbK8lM3SgjEIfqCCj4FehWwGjkOBgPWqk4%2B12birPUxxdatv6WhSdxPqlcEDjOxFn4/DPHxJYvGKlff3t4slPludBzaVgiJoT7itsGBNs2w%2BaODU2'

# Set how many images to download
for i in range(0, 10):
    
    # Download web page
    page = requests.get(firstSearch)

    # 200 status code indicates successful downloading
    print(page.status_code)

    # Initialize BeautifulSoup to parse web page
    soup = BeautifulSoup(page.content, 'html.parser')

    # Get the current aircraft image
    image = soup.find("meta", property="og:image")
    # print(image)
    print(image["content"] if image else "No meta image url given")

    # Get the next page url and load the next page
    nextPage = soup.find("div", {"class": "pdcp-pager pdcp-pager-next"})
    # print(nextPage)
    new_url = 'https://www.airliners.net'
    new_url = new_url + str(nextPage.find('a')['href'])
    print("Next page:", new_url)

    # Create image name
    imageName = str('AC-A321-' + str(i + 1) + '.jpg')

    # Download the image
    r = requests.get(image["content"])
    with open(imageName, 'wb') as outfile:
        outfile.write(r.content)

    # Move onto the next url
    firstSearch = new_url
