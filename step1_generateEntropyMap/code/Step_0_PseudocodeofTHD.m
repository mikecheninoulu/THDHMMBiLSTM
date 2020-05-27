
timestep
%%%draw the maps
for timestep = 1 to T do
    
    if timestep == 1 
        calculate statecountMap;
        gesturelist = all gesture classifications
        %make a tmp dic to cluster the gesture
         while not(all gesutre get clustered) do 
            Find gesture with min entropy as baseline
            for gesutre in gesturelist do
                if entropy(gesture,baseline gesture)<threshold
                    record as cluster flag
                end
            end
         end
    else
        calculate statecountMap;
        C = last step cluster number
        % interate for each cluster in last state
        for cluster = 1 to C 
            gesturelist = gesture in that cluster
            %make a tmp dic to cluster the gesture
             while not(all gesutre get clustered) do 
                Find gesture with min entropy as baseline
                for gesutre in gesturelist do
                    if entropy(gesture,baseline gesture)<threshold
                        record as cluster flag
                    end
                end
             end    
        end
    end
end