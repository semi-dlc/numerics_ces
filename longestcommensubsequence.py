#getestet mit Windows 10, Python 3.9.5 64-bit, Juli 2023

def lcs_matrix(s1, s2):
        
    l2 = [0] * (len(s2) + 1)
    l1 = []
    for i in range(len(s1) + 1):
        l1.append(list(l2))
    max1 = 0 #maximale Länge der größten gemeinsamen Teilsequenz
    for i in range(1, len(s1) + 1):
            #print(i)
            for j in range(1, len(s2) + 1 ):
             #   print(j)
              #  print(l1)

                if s1[i - 1] == s2[j - 1]:
                    temp = l1[i - 1][j - 1] + 1
                else:
                    temp = max(l1[i][j - 1], l1[i - 1][j])
                if max1 < temp:
                    max1 = temp

                if s1[i - 1] == s2[j - 1]:
                    l1[i][j] = l1[i - 1][j - 1] + 1
                else:
                    l1[i][j] = max(l1[i][j - 1], l1[i - 1][j])
            l1[0] = [0] * (len(s2) + 1)

    '''
    if len(s1) < len(s2):
        s1,s2 = s2, s1
    l2 = [0] * (len(s2) + 1)
    l1 = []
    for i in range(len(s1) + 1):
        l1.append(l2)

    max1 = 0

    for i in range(1, len(s1) + 1):
        print(i)
        for j in range(1, len(s2) + 1 ):
            print(j)
            print(l1)

            if s1[i - 1] == s2[j - 1]:
                temp = l1[i - 1][j - 1] + 1
            else:
                temp = max(l1[i][j - 1], l1[i - 1][j])
            if max1 < temp:
                max1 = temp

            if s1[i - 1] == s2[j - 1]:
                l1[i][j] = l1[i - 1][j - 1] + 1
            else:
                l1[i][j] = max(l1[i][j - 1], l1[i - 1][j])
        l1[0] = [0] * (len(s2) + 1)

    for i in l1:
        print(i)
    return l1

    for i in l1:
        print(i)
    '''
    return l1 


def lcs(s1, s2):
    

    # constructing lcs backwards
    ret = []
    i = len(s1) 
    j = len(s2) 
    
    l1 = lcs_matrix(s1, s2)

    start = l1[i][j] #rechts unten anfangen und pfeilen folgen
    #print(s1)
    #print(s2)
    while start != 0 and i >= 0 and j >= 0: 
        '''
        print("i, j")
        print(i, j) #i,j iterate from s1,s2.len to 1
        print("s1,s2[i]")
        print(s1[i-1], s2[j-1])
        print("l1, and diag + 1, left and up")
        print(l1[i][j])
        print(l1[i - 1][j - 1] + 1)
        print(l1[i][j - 1])
        print(l1[i - 1][j])
        '''
        if l1[i][j] == l1[i - 1][j - 1] + 1 and s1[i-1] == s2[j-1]: #l1 has one more row and more line, s1 and s2 not.
            #print("diag")
            start = l1[i][j]
            i = i - 1
            j = j - 1
            ret.append(s1[i])

        elif l1[i][j] == l1[i-1][j] :#and s1[i - 1] != s2[j - 1]:
            #print("up")
            i -= 1
        
        elif l1[i][j] == l1[i][j-1] :#and s1[i - 1] != s2[j - 1]:
            #print("left")
            j -= 1

        
        else:
            #print("Shouldn't happen. While reconstructing LCS")
            return False

    ret.reverse()
    return ret


# main

#a)
string1 = "deutschland" #across the y axis of the matrix, == i
string2 = "niedeawdde" #across the x axis of the matrix, == j

#b)
string11 = "Professor Smart ist ein viel beachteter Wissenschaftler mit einigem Erfolg auf dem Bereich der Datenstrukturen und Algorithmen. Er ist auch bei Studierenden und der Fachschaft hoch geschätzt. Er stellt stets eine faire Klausur auf."
string22 = "Im Studium gibt es viel zu tun. Immer die Hausaufgaben zu machen kann eine einfache Strategie sein, um Erfolg bei der ein oder anderen Klausur zu haben."
#satzzeichen werden als Teil des Wortes wie laut Vorgabe betrachtet
string11 = string11.split(' ')
string22 = string22.split(' ')

'''
f = open("a1.log", "rt")
g = open("a2.log", "rt")
string3 = f.read()
string4 = g.read()


#a)
list1 = [0,1,3,4,2,4,3,4]
list2 = [1,4,2,4,2,4,2,3,4]
#print(string11)
#print(string22)
'''

print(lcs(string11, string22))
