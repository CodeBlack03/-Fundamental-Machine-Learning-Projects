from model import KMeans
from utils import get_image, show_image, save_image, error
import matplotlib.pyplot as plt

def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    clusters = [2,5,10,20,50]
    errors=[]
    for c in clusters:
        num_clusters = c  # CHANGE THIS
        kmeans = KMeans(num_clusters)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        clustered_image = kmeans.replace_with_cluster_centers(image)
        
        # reshape image
        print(f"CLUSTER{num_clusters}")
        # Print the error
        
        e = error(image, clustered_image)
        errors.append(e)
        print('MSE:', e)
        image_clustered = clustered_image.reshape(img_shape)

        # show/save image
        # show_image(image)
        
        save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')
    print(errors)
    fig = plt.figure(figsize = (10, 5))
    plt.bar(clusters, errors, color ='maroon',width = 0.4)
    plt.xlabel("Number of Clusters")
    plt.ylabel("MSE Loss")
    plt.title("MSE loss with respect to num_clusters")
    plt.savefig("MSE_loss")
    plt.show()
        



if __name__ == '__main__':
    main()
