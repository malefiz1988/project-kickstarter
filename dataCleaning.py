def cat2name(cat):
    replaceString = '}{"'
    for c in replaceString: cat = cat.replace(c,"")
    dic = {}
    d = cat.split(",")
    for i in d: dic[i.split(":")[0]] = i.split(":")[1]
    return dic["name"]

def cat2slug(cat):
    replaceString = '}{"'
    for c in replaceString: cat = cat.replace(c,"")
    dic = {}
    d = cat.split(",")
    for i in d: dic[i.split(":")[0]] = i.split(":")[1]
    return dic["slug"]

def catCleaner(df):
    name = df.category.apply(cat2name)
    name.name = "category_name"
    slug = df.category.apply(cat2slug)
    slug.name = "category_slug"
    df.drop("category", inplace=True, axis =1)
    df = pd.concat([df, name, slug], axis= 1)
    return df
