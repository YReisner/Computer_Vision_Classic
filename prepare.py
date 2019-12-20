

def prepare(data, labels, param):
    # preparing the data
    # :Param
    org_data = list(zip(labels, data))  # Creating a list of the labels and it suitable data
    sift_dict = {} # creating an empty dictionary of sifts
    # for every label and data computing sifts
    for label, img in org_data:
        sift = cv2.xfeatures2d.SIFT_create()
        step_size = param['step'] #using the step size as defined in params
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
        points, sifts = sift.compute(img, kp)
        if label in sift_dict:
            sift_dict[label].append(sifts)
        else:
            sift_dict[label] = [sifts]

        all_sifts_array = list(sift_dict.values())[0]
        for value in list(sift_dict.values())[1:]:
            all_sifts_array = np.append(all_sifts_array, value, axis=0)

        model = MiniBatchKMeans(n_clusters=300, random_state=42)

        kmeans = model.fit(all_sifts_array)

        label_vec = []
        hist_vec = []
        hist_dict = {}
        for label, img in org_data:
            sift = cv2.xfeatures2d.SIFT_create()
            step_size = 6
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in
                  range(0, img.shape[1], step_size)]
            points, sifts = sift.compute(img, kp)
            img_predicts = kmeans.predict(sifts)
            img_hist, bin_size = np.histogram(img_predicts, bins=300)
            hist_vec.append(img_hist)
            label_vec.append(label)
            img_hist = img_hist.reshape(1, 300)
            if label in hist_dict:
                hist_dict[label] = np.append(hist_dict[label], img_hist, axis=0)
            else:
                hist_dict[label] = img_hist

        return hist_vec, label_vec
def make_hist(data)


"""
path = r'C:\Users\BIGVU\Desktop\Yoav\University\101_ObjectCategories'
#path = r'C:\Users\gilei\Desktop\comp\Computer_Vision_Classic-master\101_ObjectCategories'
files = os.listdir(path)[0:3]

labels = []
img_list = []
for file in files:
    newPath = path+"\\"+file
    for img in glob.glob(newPath+"\\*.jpg")[0:20]:
        raw = cv2.imread(img, 0)
        im_data = np.asarray(raw)
        #gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        sized = cv2.resize(im_data,(100,100))
        img_list.append(sized)
        labels.append(file)
array = np.array(img_list)
"""

org_data = list(zip(labels, array))
sift_dict = {}
for label, img in org_data:
     sift = cv2.xfeatures2d.SIFT_create()
     step_size = 6
     kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
     points, sifts = sift.compute(img, kp)
     if label in sift_dict:
         sift_dict[label] = np.append(sift_dict[label],sifts,axis=0)
     else:
        sift_dict[label] = sifts






svm = SVC(C=1.0, kernel='linear',gamma='scale')
svm.fit(hist_vec,label_vec)