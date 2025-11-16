#adjazenzliste eines UNgerichteten Graphen, welcher in Adjazenzliste freunde gespeichert ist
#ist freunde ein bipartiter Graph?
#freunde bipartit <-> keine ungerade Zykel

#annahme: freunde von freunde gelten nicht als freunde - beziehung freund ist nicht transistiv
#weitere Annahme: Man ist nicht sein eigener Freund - Beziehung Freund ist nicht reflexiv

#Annahme zum Format: Adjazenzliste
#aufgabe: Nutze Breitensuche um Länge von Zykel zu bestimmen
#2 buckets/rooms

WHITE = 42
GRAY = 43
BLACK = 44

COLOR1 = 1 #raum1
COLOR2 = 0 #raum2
COLOR_now = 0

color_search = []
color_b = []

def bfs(adj[n], start, color_b, color_search):
    color_b[start] = GRAY
    wait = []
    wait.enqueue(start)
    while not wait.isEmpty():
        v = wait.dequeue()
        for w in range(0, adj[v]): #Schleife, die über alle Elemente in einem einzelnen Eintrag einer Adjazenzliste iteriert?
            if color_b[w] == WHITE:
                color_b[w] = GRAY
                color_search[w] = COLOR_now
                wait.enqueue(w)
            elif color_b[w] == BLACK: #GRAY wird aufgrund Nichtreflexivität der Beziehung Freund nicht behandelt
                #Zykel gefunden
                if color_search[w] == color_search[v]: #zwei benachbarte Knoten würden durch ihre Zuteilung einen Konflikt erzeugen
                    return -1
        COLOR_now = (COLOR_now + 1) % 2         
        color_b[v] = BLACK
    return 0

def completeBFS(adj[n], n):
    color_b[n] = WHITE
    iterator = 0
    for i in range(0,n):
        if color_b[i] == WHITE:
            if bfs(adj, i, color_b, color_search) == -1:
                return -1


def aufteilung(freunde):
    if completeBFS(freunde, len(freunde)) == -1:
        return False
    return True

            

