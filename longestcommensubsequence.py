def lcs(s1, s2):
    l2 = [0] * (len(s2) + 1)
    l1 = []
    for i in range(len(s1) + 1):
        l1.append(l2)

    max1 = 0

    for i in range(1, len(s1) + 1):
        print(i)
        temp = 0
        for j in range(1, len(s2) + 1): #the problem arises somewhere here. the result of the last row is returned here every time, god knows why
            print(j)
            print(l1)
            if s1[i - 1] == s2[j - 1]:
                temp = l1[i - 1][j - 1] + 1
            else:
                temp = max(l1[i][j - 1], l1[i - 1][j])
            if max1 < temp:
                max1 = temp
            l1[i][j] = temp
            

   #l1[0] = [0] * (len(s2) + 1)

    for i in l1:
        print(i)

    # constructing lcs backwards
    ret = []
    i = len(s1)
    j = len(s2)
    start = l1[i][j]
    while start != 0 and i > 0 and j > 0:
        if l1[i][j] == l1[i - 1][j - 1] + 1 and s1[i - 1] == s2[j - 1]:
            start = l1[i][j]
            i = i - 1
            j = j - 1
            ret.append(s1[i])
        elif l1[i][j] == l1[i - 1][j] and s1[i - 1] != s2[j - 1]:
            i = i - 1
        elif l1[i][j] == l1[i][j - 1] and s1[i - 1] != s2[j - 1]:
            j = j - 1
        else:
            print("Shouldn't happen. While reconstructing LCS")
            return False

    ret.reverse()
    return ret


# main
string1 = "vidra" #really weird and buggy
string2 = "nitter"
print(lcs(string1, string2))
