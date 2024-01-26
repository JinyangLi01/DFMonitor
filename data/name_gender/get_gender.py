import pandas as pd
import requests, json


def getGenders(names):
    url = ""
    cnt = 0
    if not isinstance(names, list):
        names = [names, ]

    for name in names:
        if url == "":
            url = "name[0]=" + name
        else:
            cnt += 1
            url = url + "&name[" + str(cnt) + "]=" + name

    req = requests.get("https://api.genderize.io?" + url)
    results = json.loads(req.text)

    retrn = []
    print(results)
    for result in results:
        if result["gender"] is not None:
            retrn.append((result["gender"], result["probability"], result["count"]))
        else:
            retrn.append((u'None', u'0.0', 0.0))
    return retrn


if __name__ == '__main__':
    names = pd.read_csv("baby-names.csv")
    print(names.head())

    names["prediction"] = names.apply(lambda row: getGenders(row["name"]), axis=1)


    print(names.head())