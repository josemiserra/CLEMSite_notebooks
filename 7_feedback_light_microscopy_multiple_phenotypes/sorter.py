from scipy.spatial import KDTree

def listOfClosest(list_sem, radius = 100):
        """
         Gives back the indices of elements to be removed

        Notice that this method requires some sophistication.
        In this approach we just leave the first element we find.
        :param m_list_px:
        :param m_list_py:
        :param radius:
        :return:
        """
        cp_list = list_sem.copy()
        # for the tree
        ncp_list = list_sem.copy()
        ncp_list_x = ncp_list[:, 0]
        ncp_list_y = ncp_list[:, 1]
        tree_list = list(zip(ncp_list_x, ncp_list_y))
        list_excluded = []
        for ind,el in enumerate(cp_list):
            el = el[0:2]
            distances_p, indexes =  KDTree(tree_list).query(el,k=2) # closest element that is not yourself
            if( distances_p[1] < radius):
                list_excluded.append(ind)
                ncp_list = np.delete(ncp_list,[ind],axis=0)
                ncp_list_x = ncp_list[:, 0]
                ncp_list_y = ncp_list[:, 1]
                tree_list = list(zip(ncp_list_x, ncp_list_y))
                if(len(tree_list)<2):
                    break

        return list_excluded