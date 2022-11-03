
import asyncio
from ttictoc import tic,toc
from txtai.pipeline import Labels

labels = Labels()
result=None


tic()

async def foo(text,tags):
    print(text)
    await asyncio.sleep(.001)
    result=tags[labels(text, tags)[0][0]]
    print(result)
    return result
    
async def manager():
    tasklist=[]
    
    data = ["Dodgers lose again, give up 3 HRs in a loss to the Giants",
            "Giants 5 Cardinals 4 final in extra innings",
            "Dodgers drop Game 2 against the Giants, 5-4",
            "Flyers 4 Lightning 1 final. 45 saves for the Lightning.",
            "Slashing, penalty, 2 minute power play coming up",
            "What a stick save!",
            "Leads the NFL in sacks with 9.5",
            "UCF 38 Temple 13",
            "With the 30 yard completion, down to the 10 yard line",
            "Drains the 3pt shot!!, 0:15 remaining in the game",
            "Intercepted! Drives down the court and shoots for the win",
            "Massive dunk!!! they are now up by 15 with 2 minutes to go"]

    tags = ["Baseball", "Football", "Hockey", "Basketball"]
    
    
    for i in data:
        task=asyncio.create_task(foo(str(i),tags))
        tasklist.append(task)
    
    data=asyncio.gather(*tasklist)
    print('Result::')
    result=await data
    print('Done')
    return result
    


gdb=asyncio.run(manager())
timeinfo=toc()
print(f'{gdb} --{timeinfo}')
