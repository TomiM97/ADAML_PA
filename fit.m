function weights = fit(model)
 
    pseudox = kernelRBF(model);
    if model.regrType == 3
        weights                  = linsolve(pseudox, model.Y);
    else
        if model.intercept == 1
             weights = kpls(pseudox, model.Y - mean(model.Y), model.regrType, model.dim);
        else
             weights = kpls(pseudox, model.Y, model.regrType, model.dim);
        end
    end
    
end