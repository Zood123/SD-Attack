
import numpy as np



def cal_scores_D(A, eig_vals, eig_vec, filtered_edges, target_node, eig_vec2, inverse_u, degrees):
    k = len(eig_vals) # set k to smaller values if too many eigenvalues are used
    filtered_edges = [ [edge[0],edge[1]] for edge in filtered_edges]
    inverse_u = inverse_u[:k]
    org_y = np.power(eig_vec2[target_node,:k], 2)


    A[A > 1] = 1
    return_values = []

    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]

        # calculate the change of eigenvalues
        eig_vals_res = (1 - 2 * A[filtered_edge[0], filtered_edge[1]]) * (
                2 * eig_vec[filtered_edge[0], :] * eig_vec[filtered_edge[1], :] - eig_vals *
                (np.square(eig_vec[filtered_edge[0], :]) + np.square(eig_vec[filtered_edge[1], :])))

        top_eigvals = eig_vals[:k]
        new_eigenvals = eig_vals + +eig_vals_res

        # calculate the change of eigenvectors
        delta_ = -(np.power(degrees[target_node], 0.5)) * (1 - 2 * A[filtered_edge[0], filtered_edge[1]]) * ( 
                    inverse_u[:, filtered_edge[0]] * (eig_vec[filtered_edge[1], :k] - top_eigvals * eig_vec[filtered_edge[0], :k]) + inverse_u[:,filtered_edge[1]] * (
                                eig_vec[filtered_edge[0], :k] - top_eigvals * eig_vec[filtered_edge[1], :k]))

        new_y = np.power(eig_vec2[target_node,:k]+delta_,2)



        # sort the order of eigenvalues and corresponding eigenvectors
        for i in range(len(new_eigenvals[:k])-1):
            if new_eigenvals[i] > new_eigenvals[i+1]:
                temp_val = new_eigenvals[i+1]
                temp_vec = new_y [i+1]

                new_eigenvals[i+1] = new_eigenvals[i]
                new_y[i+1] = new_y[i]
                new_eigenvals[i] = temp_val
                new_y[i] = temp_vec

        # Start to calculate the w1 distance
        new_pos = 0
        area = 0
        for i,eig_val in enumerate(eig_vals[:k]):

            current_load = org_y[i]
            if new_pos == len(eig_vals[:k]):
                break

            while(1):
                if new_pos == len(eig_vals[:k]):
                    break
                delta_x = abs(eig_val - new_eigenvals[new_pos])
                if  current_load > new_y[new_pos]:
                    area = area + delta_x*new_y[new_pos]
                    current_load = current_load -new_y[new_pos]
                    new_pos = new_pos + 1
                else:
                    area = area + delta_x*current_load
                    new_y[new_pos] = new_y[new_pos] - current_load
                    break

        return_values.append(area)

        print("The scoring progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
    print("\n")

    return np.array(return_values)
